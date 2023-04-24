import numpy as np
import torch

def wavelentgh2cords(wv, lambda0 = 340, lambda1 = 2500):
    """
    transforma una longitud de onda a una coordenada en el espacio de las longitudes de onda
    """
    return (wv-lambda0)/(lambda1-lambda0)*2-1


def coords2wavelentgh(x, lambda0 = 340, lambda1 = 2500):
    """
    transforma una coordenada en el espacio de las longitudes de onda a una longitud de onda
    """
    return (x+1)/2*(lambda1-lambda0)+lambda0

def escalon_unitario(x):
    #return torch.heaviside(x,torch.zeros(1))
    return torch.where(x>=0, 1, 0)

def pasa_banda(x, epsilon):
    """
    filtro pasa banda centrado en 0 de tamaño epsilon usando el escalon_unitario
    """

    return 1*(escalon_unitario(x+epsilon/2) - escalon_unitario(x-epsilon/2))
    
def tren_impulsos(x,  spectral_res, epsilon):
    """
    filtro tren de impulsos infinito centrado en 0 de tamaño epsilon usando el filtro pasa banda
    """

    #grid = torch.linspace(-epsilon,epsilon,2*epsilon)

    """
    Center the module of x in the interval [-spectral_res/2, spectral_res/2]
    """
    mod = torch.remainder(x, spectral_res)

    return pasa_banda(mod, epsilon) 




if __name__=="__main__":
    L = 99
    xmin = -1
    xmax = 1
    x = torch.linspace(xmin,xmax,4*L)

    y = escalon_unitario(x)#.numpy()

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    #axs.plot(x.numpy(), y.numpy(), "D", label = "escalon unitario")

    spectral_res = 2*(xmax - xmin)/L
    y = pasa_banda(x, spectral_res)#.numpy()

    axs.plot(x.numpy(), y.numpy(), "o", label = "pasa banda")

    y = tren_impulsos(x, spectral_res = spectral_res, epsilon = (1/4)*spectral_res)#.numpy()

    axs.plot(x.numpy(), y.numpy(), "*-", label = "tren de impulsos")

    a = spectral_res*np.linspace(-1,1,100)
    axs.plot(a, np.ones_like(a), "rD-", label = "spectral resolution", markersize = 1)

    #b = (1/4)*spectral_res*np.linspace(-1,1,100)
    #y = torch.remainder(x, spectral_res)
    axs.plot(x, y, "ko-", label = "epsilon", markersize = 1)
    axs.legend()
    plt.show()