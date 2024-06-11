import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft


#Recibe señal(arreglo) => devuelve potencia
def potencia(x):
    p=0
    for i in range(len(x)):
        p+=np.power(np.abs(x[i]),2)
    return p/len(x)

#Recibe señal(arreglo) => Devuelve función de autocorrelación
def auto_corr(y):
    N=len(y)
    r=np.zeros(N)
    for k in range(N-1):
       for i in range(N-k):
            r[k] += y[i]*y[i+k]
    r=r/N
    return r

#Estimador de periodograma
#Recibe señal(arreglo) y parametro para realizar la fft de la señal
#=> Devuelve periodograma
def Periodograma(y,N_fft):
    N=len(y)
    psd=np.power(np.abs(fft(y,N_fft)),2)
    #psd=psd/N
    return psd/N

#Recibe señal(arreglo), porcentaje de solapamiento(pwe_margin), tipo de ventana(wind), tamaño de ventana M
def welch_metod_PSD(x,per_margin,wind,M):
    N=len(x)
    K=int(M*(1-per_margin/100)) #Desplazamiento
    L=int(N/K) #Cantidad de segmetos

    v=signal.get_window(wind,M)
    V=potencia(v) #Calculo la potencia de la ventana

    S_w=np.zeros(N)
    for i in range(L-1):  
        x_i=x[i*K : (i*K + M)]*v
        S_w+=Periodograma(x_i,N)/V
    return S_w/L

#Recibe señal, retorna señal mapeada 1 bit a 1 simbolo
def mapeo(x):
    y=2*x-1
    return y

#Recibe señal 
def canal_discreto(x,h):
    g=signal.lfilter(h,[1],x)
    y=g + np.random.normal(0,0.002,len(g))
    return y

def retardo(x,D):
    ret = np.zeros(D+1)
    ret[D] = 1
    return signal.lfilter(ret,[1],x)


def algoritmo_LMS(y,d,paso,M):

#                                                  d(n)
#                                                   |
#             +-------------+                       |
#             |             |                       +
#   y(n) ---> |      w      | ----> d_moño(n) --> -    ---> e(n)
#             |             |
#             +-------------+ 

    d_moño=np.zeros(len(y))
    error=np.zeros(len(y))
    w=np.zeros((M,len(y)))
    for i in range(M, len(d)-M):
        ventana = y[i:i-M:-1]
        d_moño[i-M] = np.dot(w[:,i],ventana)
        error[i-M] = d[i] - d_moño[i-M]
        w[:,i+1] = w[:,i] + paso*error[i-M]*ventana
    return w, error, d_moño

def ej_1():
    #a)
    b=np.random.binomial(1,0.5,2000)
    x=mapeo(b)
    plt.stem(x)
    plt.xlim(0,100)
    plt.title('Señal x(n)')
    plt.xlabel('n')
    plt.ylabel('x(n)')
    plt.grid()
    plt.show()

    #b)
    h=[0.5,1,0.2,0.1,0.05,0.01]
    y=canal_discreto(x,h)
    N=np.linspace(0,2000,len(x))
    plt.stem(y)
    plt.xlim(0,100)
    plt.title('Salida del canal discreto y(n)')
    plt.xlabel('n')
    plt.ylabel('y(n)')
    plt.grid()
    plt.show()

    #c)
    R_x_teo=[1]
    R_x=auto_corr(x)
    R_y=auto_corr(y)
    plt.xlim(0,100)
    plt.plot(N,R_x,label='$R_x$')
    plt.plot(N,R_y,label='$R_y$')
    plt.ylabel('r(k)')
    plt.xlabel('k')
    plt.title('Autocorrelación')
    plt.grid()
    plt.legend()
    plt.show()
    
    S_x=welch_metod_PSD(x,50,'hamming',80)
    S_x_teo=fft(R_x_teo,len(S_x))
    S_y=welch_metod_PSD(y,50,'hamming',80)
    w=np.linspace(0,2*np.pi,len(S_x))
    plt.plot(w,10*np.log10(S_x),label='$S_x$ - Welch')
    plt.plot(w,10*np.log10(S_y),label='$S_y$ - Welch')
    plt.plot(w,10*np.log10(S_x_teo),label='$S_x$ - Teórica')
    plt.xlabel('w [rad/s]')
    plt.ylabel('PSD [dB]')
    plt.title('PSD')
    plt.legend()
    plt.xlim(0,np.pi)
    plt.grid()
    plt.show()

def ej_2():
    #Parametros
    D=9
    realizaciones=500
    mu=0.05
    orden=8
    N=1000
    N_i=np.linspace(0,N,N)
    p=0.5
    h=[0.5,1,0.2,0.1,0.05,0.01]
    labels_1 = ['D = 1','D = 2','D = 3','D = 4','D = 5','D = 6','D = 7','D = 8','D = 9']
    
    Jlim_inf=300
    Jlim_sup=900
    J_infinito=np.zeros(D)
    for i in range(1,D+1):
        J=np.zeros(N)
        for k in range(realizaciones):
            d=mapeo(np.random.binomial(1,p,N))
            y=canal_discreto(d,h)
            w, e, d_moño = algoritmo_LMS(y,retardo(d,i),mu,orden)
            J = J + np.power(np.abs(e),2)
        J = J/realizaciones
        J_infinito[i-1]=np.sum(J[Jlim_inf:Jlim_sup])/(Jlim_sup-Jlim_inf)
        plt.plot(N_i,J,label=labels_1[i-1])
    plt.legend()
    plt.xlabel('Nro de iteración')
    plt.ylabel('J(n)')
    plt.title('Curvas de Aprendizaje según retardo D')
    plt.show()

    n=np.linspace(0,D,len(J_infinito))
    plt.scatter(n,J_infinito,marker='x')
    plt.ylabel('J($\infty$)')
    plt.xticks(n,labels_1)
    plt.title('J($\infty$) según retardo D')
    plt.yscale('log')
    plt.grid()
    plt.show()

    D_min=np.argmin(J_infinito)+1
    d=mapeo(np.random.binomial(1,p,N))
    y=canal_discreto(d,h)
    d_ret=retardo(d,D_min)
    w, e, z=algoritmo_LMS(y,d_ret,mu,orden)
    plt.title('Señal original desplazada x($n-D_{opt}$) vs Secuencia equializada z(n)')
    plt.stem(retardo(z,orden),'r',label='z(n)')
    plt.stem(d_ret,'g',label='x($n-D_{opt}$)')
    plt.legend()
    plt.show()

    conv_w_h=np.convolve(w[:,980],h)
    plt.stem(conv_w_h)
    plt.ylabel('h$\circledast$w')
    plt.xlabel('n')
    plt.title('Respuesta impulsiva')
    plt.grid()
    plt.show()

    plt.plot(np.linspace(0,2*np.pi,N),fft(conv_w_h,N))
    plt.xlabel('w [rad/s]')
    plt.ylabel('$\mathcal{F}$(h$\circledast$w)')
    plt.title('Respuesta en frecuencia')
    plt.grid()
    plt.show()
    
    
        


ej_1()
ej_2()
