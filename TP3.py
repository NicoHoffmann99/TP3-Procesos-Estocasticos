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
    g=np.convolve(x,h,'same')
    y=g + np.random.normal(0,0.002,len(g))
    return y
'''
#                                                  d(n)
#                                                   |
#             |-------------|                       |
#             |             |                       +
#   y(n) ---> |      w      | ----> y_moño(n) -->  -   ---> e(n)
#             |             |
#             |-------------|
'''
'''
def algoritmo_LMS(y,d,mu,M):
    #Semilla w del algoritmo
    w=np.zeros(M)
    e=np.zeros(len(d))
    y_moño=np.zeros(len(d))

    for i in range(M, len(y)):
        #Extraigo porción de y para calcular y_moño
        y_porción=y[i:(i-M+1)]
        y_moño[i]=np.dot(w,y_por)

        e[i] = d[i] -y_moño[i]

        w = w + mu*e[i]*y_porción
    
    return w, e, y_moño
'''
    

def ej_1():
    #a)
    b=np.random.binomial(1,0.5,1000)
    x=mapeo(b)
    plt.stem(x)
    plt.xlim(0,100)
    plt.show()

    #b)
    h=[0.5,1,0.2,0.1,0.05,0.01]
    y=canal_discreto(x,h)
    N=np.linspace(0,1000,len(x))
    plt.stem(y)
    plt.xlim(0,100)
    plt.show()

    #c)
    R_x=auto_corr(x)
    R_y=auto_corr(y)
    plt.plot(N,R_x)
    plt.plot(N,R_y)
    plt.show()

    S_x=welch_metod_PSD(x,50,'hamming',80)
    S_y=welch_metod_PSD(y,50,'hamming',80)
    plt.plot(N,S_x)
    plt.plot(N,S_y)
    plt.show()


ej_1()