
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from sensingmod import tren_impulsos
import torch

def snr2std(x, snr):
    m = np.prod(x.shape[:]).astype("float").tolist()
    divisor = m*10**(snr/10)
    norm = torch.pow(torch.norm(x), 2)
    stddev = torch.sqrt(torch.divide(norm, divisor)).cpu()
    return stddev.detach().numpy().tolist()

class Sensor():
    def __init__(self,d=0,epsilon=0,mean=0,snr=0,device=None,function=None):
        #self.wv0=wv0
        #self.wv1=wv1
        self.d=d
        self.epsilon=epsilon
        self.mean=mean
        self.snr=snr
        self.device=device
        self.function=function
        # Noise generation

    def __call__(self,intensities, coords):
        if self.snr>=1:
            std = snr2std(intensities, self.snr)
            coords = coords[:, -1:].to(self.device)
            n = torch.normal(self.mean,std,(coords.shape[0],1)).to(self.device)
            Rsensor = intensities.to(self.device)+n.to(self.device)*self.function(coords.to(self.device))
        else:
            std = snr2std(intensities, self.snr)
            coords = coords[:, -1:].to(self.device)
            Rsensor = intensities.to(self.device)

        
        #condicion_range = torch.logical_and(coords > self.wv0, coords < self.wv1).to(Rsensor.dtype)
        #condicion_range = torch.where(condicion_range, torch.ones_like(condicion_range), - torch.ones_like(condicion_range))
        #condicion_range = coordinates>0
        #tmp = condicion_range*((Rsensor+1)/(2))
        #Rsensor = tmp*2-1
        return Rsensor.to(self.device)
    
def bandselector(coords, wv0, wv1):
    coords = coords[:, -1]
    condicion_range = torch.logical_and(coords >= wv0-0.1, coords <= wv1+0.1)
    return condicion_range

