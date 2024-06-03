import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft

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
    y=np.zeros(len(x))
    for i in range(len(x)):
        y[i]=x[i]
        if x[i]==0:
            y[i]=-1
    return x

#Recibe señal 
def canal_discreto(x,h):
    g=np.convolve(x,h)
    y=g + np.random.normal(0,0.002,len(g))
    return y

def ej_1():
    #a)
    b=np.random.binomial(1,0.5,1000)
    x=mapeo(b)

    h=[0.5,1,0.2,0.1,0.05,0.01]
    y=canal_discreto(x,h)
    N_1=np.linspace(0,1000,len(x))
    N_2=np.linspace(0,1000,len(y))
    plt.plot(N_1,x)
    plt.show()
    plt.plot(N_2,y)
    plt.show()


ej_1()