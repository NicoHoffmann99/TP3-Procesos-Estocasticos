import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft
from scipy import stats
from scipy import special

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

#Recibe señal de entrada, devuelve salida del canal discreto(convolucion con rta impulsiva y ruido)
def canal_discreto(x,h,var):
    g=signal.lfilter(h,[1],x)
    v=np.random.normal(0,np.sqrt(var),len(g))
    y=g + v
    return y

#Recibe señal, devuelve señal con retardo D(entero) aplicado
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


def umbral_optimo(x1,x0):
    umbral=((pow(x1,2)-pow(x0,2)))/(2*(x1-x0))
    return umbral

def clasificador_ML(x, umbral, amplitud):
    x_clasificado=np.zeros(len(x))
    for i in range(len(x)):
        if x[i]<umbral:
            x_clasificado[i]=-1
        if x[i]>umbral:
            x_clasificado[i]=1
        if x[i]==umbral:
            x_clasificado[i]=mapeo(np.random.binomial(1,0.5,1))*amplitud
    return x_clasificado*amplitud

def obtener_SER(x,x_clasificado):
    cant_errores=0
    for i in range(len(x)):
        if x[i]!=x_clasificado[i]:
            cant_errores=cant_errores+1
    SER_coef=cant_errores/len(x)
    return SER_coef

def funcion_Q(x,var):
    Q=0.5-0.5*special.erf((x/np.sqrt(var))/np.sqrt(2))
    return Q

def pdf_normal(mu,var,long):
    sigma = np.sqrt(var)
    n = np.linspace(mu-4*sigma, mu+4*sigma, long)
    x = stats.norm.pdf(n,mu,sigma)
    return n, x

def ej_1():
    h=[0.5,1,0.2,0.1,0.05,0.01]
    p=0.5
    N=1000
    var=0.002
    
    #a)
    b=np.random.binomial(1,p,N)
    x=mapeo(b)
    plt.stem(x)
    plt.xlim(0,100)
    plt.title('Señal x(n)')
    plt.xlabel('n')
    plt.ylabel('x(n)')
    plt.grid()
    plt.show()

    #b)
   
    y=canal_discreto(x,h,var)
    N=np.linspace(0,1000,len(x))
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
    
    S_x=welch_metod_PSD(x,5,'hamming',10)
    S_x_teo=fft(R_x_teo,len(S_x))
    S_y=welch_metod_PSD(y,5,'hamming',10)
    
    w_prim, S_y_1=signal.freqz(h,[1],len(x))
    S_y_2=fft([0.002],len(S_y_1))
    S_y_teo=np.power(np.abs(S_y_1),2)*S_x_teo+S_y_2
    
    w=np.linspace(0,2*np.pi,len(S_x))
    plt.plot(w,10*np.log10(S_x),label='$S_x$ - Welch')
    plt.plot(w,10*np.log10(S_y),label='$S_y$ - Welch')
    plt.plot(w,10*np.log10(S_x_teo),label='$S_x$ - Teórica')
    plt.plot(w_prim,10*np.log10(S_y_teo),label='$S_y$ - Teórica')
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
    var=0.002
    h=[0.5,1,0.2,0.1,0.05,0.01]
    labels_1 = ['D = 1','D = 2','D = 3','D = 4','D = 5','D = 6','D = 7','D = 8','D = 9']
    
    Jlim_inf=300
    Jlim_sup=900
    J_infinito=np.zeros(D)
    for i in range(1,D+1):
        J=np.zeros(N)
        for k in range(realizaciones):
            d=mapeo(np.random.binomial(1,p,N))
            y=canal_discreto(d,h,var)
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
    y=canal_discreto(d,h,var)
    d_ret=retardo(d,D_min)
    w, e, z=algoritmo_LMS(y,d_ret,mu,orden)
    #Tomo coeficientes una vez en estado estacionario
    w_a=w[:,980]

    plt.title('Señal original desplazada x($n-D_{opt}$) vs Secuencia equializada u(n)')
    plt.stem(signal.lfilter(w_a,[1],y),'r',label='u(n)')
    plt.stem(d_ret,'g',label='x($n-D_{opt}$)')
    plt.legend()
    plt.show()


    plt.stem(w_a,'r',label='w(n)')
    plt.stem(h,'g',label='h(n)')
    plt.grid()
    plt.xlabel('n')
    plt.ylabel('h(n) & w(n)')
    plt.legend()
    plt.show()

    conv_w_h=np.convolve(w_a,h)
    plt.stem(conv_w_h)
    plt.ylabel(r'h(n)$\ast$w(n)')
    plt.xlabel('n')
    plt.title('Convolución entre respuestas impulsivas h(n) y w(n)')
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(np.linspace(0,2*np.pi,N),fft(w_a,N),label='$\mathcal{F}$(w(n))')
    plt.plot(np.linspace(0,2*np.pi,N),fft(h,N),label='$\mathcal{F}$(h(n))')
    plt.grid()
    plt.title('$\mathcal{F}$(h(n)) vs $\mathcal{F}$(w(n))')
    plt.xlabel('w [rad/s]')
    plt.xlim(0,np.pi)
    plt.legend()
    plt.show()

    plt.plot(np.linspace(0,2*np.pi,N),np.abs(fft(w_a,N)*fft(h,N)))
    plt.ylim(0,1.1)
    plt.xlabel('w [rad/s]')
    plt.ylabel(r'|$\mathcal{F}$(h(n)$\ast$w(n))|')
    plt.title('Módulo de la respuesta en frecuencia')
    plt.grid()
    plt.xlim(0,np.pi)
    plt.show()

    plt.plot(np.linspace(0,2*np.pi,N),np.angle(fft(w_a,N)*fft(h,N)))
    plt.xlabel('w [rad/s]')
    plt.ylabel(r'$\angle$$\mathcal{F}$(h(n)$\ast$w(n))')
    plt.title('Fase de la respuesta en frecuencia')
    plt.grid()
    plt.xlim(0,np.pi)
    plt.show()

    
    
def ej_3():
    #a)
    #Parámetros
    
    vars_1=[0.1,0.2,0.3]
    h=[10*0.1]
    N_1=10000     
    p=0.5
    labels_a=['$\sigma^2$ = 0.1','$\sigma^2$ = 0.2','$\sigma^2$ = 0.3']
    colors=['r','y','g']
    amplitud_1=1
    titulos_a=['Histograma para señal con RGB de $\sigma^2$ = 0.1','Histograma para señal con RGB de $\sigma^2$ = 0.2','Histograma para señal con RGB de $\sigma^2$ = 0.3']
    for i in range(len(vars_1)):
        x=mapeo(np.random.binomial(1,p,N_1))
        u=canal_discreto(x,h,vars_1[i])
        umbral=umbral_optimo(amplitud_1,-amplitud_1)

        plt.axvline(umbral,0,1,linestyle='--',label=f'Umbral óptimo = {umbral}')
        plt.hist(u,bins=100,label=labels_a[i],color=colors[i])
        bin_ancho = (u.max() - u.min()) / 100

        n, pdf1 = pdf_normal(amplitud_1,vars_1[i],len(x))
        plt.plot(n, 0.5*pdf1 * N_1 * bin_ancho,'k',label='$f(u|H_1)$')
        n, pdf2 = pdf_normal(-amplitud_1,vars_1[i],len(x))
        plt.plot(n, 0.5*pdf2 * N_1 * bin_ancho,'b',label='$f(u|H_0)$')
        plt.xlabel('Amplitud del símbolo + ruido')
        plt.title(titulos_a[i])
        plt.grid()
        plt.legend()
        plt.show()
    
    #b)
    vars_2=0.5
    amplitudes=[1,3]
    N_2=100000
    labels_b=['$|x|$ = 1' ,'$|x|$ = 3']
    titles_b=['Histograma para x(n) de amplitud 1','Histograma para x(n) de amplitud 3']
    for i in range(len(amplitudes)):
        x=amplitudes[i]*mapeo(np.random.binomial(1,p,N_2))
        u=canal_discreto(x,h,vars_2)
        umbral=umbral_optimo(amplitudes[i],-amplitudes[i])
        print('UMBRAL: ',umbral)

        u_clasificada=clasificador_ML(u,umbral,amplitudes[i])
        SER_coef=obtener_SER(x,u_clasificada)
        print('Coeficiente SER: ', SER_coef)

        Pe=funcion_Q(amplitudes[i],vars_2)
        print('Probabilidad de error: ',Pe)

        plt.hist(u,bins=100,label=labels_b[i],color='r')
        bin_ancho = (u.max() - u.min()) / 100

        n, pdf1 = pdf_normal(amplitudes[i],vars_2,len(x))
        plt.plot(n, 0.5*pdf1 * N_2 * bin_ancho,'k',label='$f(u|H_1)$')
        n, pdf2 = pdf_normal(-amplitudes[i],vars_2,len(x))
        plt.plot(n, 0.5*pdf2 * N_2 * bin_ancho,'b',label='$f(u|H_0)$')
        plt.axvline(umbral,0,1,linestyle='--',label=f'Umbral óptimo = {umbral}',color='k')

        plt.xlabel('Amplitud del símbolo + ruido')
        plt.grid()
        plt.title(titles_b[i])
        plt.legend()
        plt.show()

    #c)
    N_3=10000
    amplitud=1
    vars_3=np.arange(5,0.0,-0.2)

    SER_array=np.zeros(len(vars_3))
    Pe=np.zeros(len(vars_3))
    SNR_array=np.zeros(len(vars_3))
    for i in range(len(vars_3)):
        x=amplitud*mapeo(np.random.binomial(1,p,N_3))
        u=canal_discreto(x,h,vars_3[i])
        umbral=umbral_optimo(amplitud,-amplitud)
        u_clasificada=clasificador_ML(u,umbral,amplitud)
        SER_array[i]=obtener_SER(x,u_clasificada) 
        Pe[i]=funcion_Q(amplitud,vars_3[i])
        SNR_array[i]=potencia(x)/potencia(np.random.normal(0,np.sqrt(vars_3[i]),len(x)))

    plt.plot(SNR_array,Pe,label='$P_e$')
    plt.plot(SNR_array,SER_array,label='$SER^3$')
    plt.title('$SER^3$ y $P_e$ en función del SNR')
    plt.xlabel('SNR')
    plt.ylabel('$SER^3$ / $P_e$')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.show()
        


    



#ej_1()
ej_2()
#ej_3()