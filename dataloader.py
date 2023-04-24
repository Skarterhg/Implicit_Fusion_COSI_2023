from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import cv2 as cv
import torch
import os
import matplotlib.pyplot as plt
from forwardmodel import Sensor, bandselector

def intensitiesofimage(int):
    int=torch.from_numpy(int).float()
    int=normalize_minus1to1(int)    
    int = torch.reshape(int,(-1,1))
    int = normalize_minus1to1(int)
    return int

def normalize_minus1to1(x):
        #x = torch.from_numpy(x).float()
        x = (x - torch.min(x))/(torch.max(x) - torch.min(x))
        x = 2*x - 1
        return x

def getcoords_andstack(img):
        imgx = torch.linspace(-1,1,img.shape[0])
        imgy = torch.linspace(-1,1,img.shape[1]) #make vectors for x,y
        imglambda = torch.linspace(-1,1,img.shape[2])

        xv, yv, lambdav = torch.meshgrid(imgx,imgy,imglambda, indexing='ij') #separate the gradient of the coordinates

        #Xcord = torch.stack((xv,yv,lambdav),axis=len(img.shape)) #Stack the coordinates 
        #Xcord = torch.reshape(Xcord,(-1,len(img.shape))) #Values of the tensor for coords

        Xcordalt = torch.stack((xv,yv,lambdav),axis=len(img.shape))#[:,:,[0, 2],: ] #Stack the coordinates 
        Xcordalt = torch.reshape(Xcordalt,(-1,len(img.shape))) #Values of the tensor for coords

        return Xcordalt

def coordinates_processing(path, debug, size = (64,64), bands = 5,s1init = None,s1end = None,s2init = None,s2end = None,mean = None,snr = None,function_to_s1 = None,function_to_s2 = None,device = None):
    img = path
    img = img.astype(np.float32)

    # sample the images channels in "bands" bands
    img = img[:, :, np.linspace(0, img.shape[-1], bands, dtype=int, endpoint=False)]
    
    if debug == True:
        size = (32,32)
    img = cv.resize(img, size)
     
    Imgs1=img
    Imgs2=img[...,0::2]

    imggt_shape = img.shape #get the shape of the image
    imgs1_shape = Imgs1.shape
    imgs2_shape = Imgs2.shape
    
    Xcordalts1 = getcoords_andstack(Imgs1)
    Xcordalts2 = getcoords_andstack(Imgs2)
    coordenadas_test = getcoords_andstack(img)

    conditions1 = bandselector(Xcordalts1, s1init, s1end)
    conditions2 = bandselector(Xcordalts2, s2init, s2end)

    Xcordalts1 = Xcordalts1[conditions1,...]
    Xcordalts2 = Xcordalts2[conditions2,...]


    Yintimgs1 = intensitiesofimage(Imgs1)
    Yintimgs2 = intensitiesofimage(Imgs2)
    Yint = intensitiesofimage(img)

    Yintimgs1 = Yintimgs1[conditions1]
    Yintimgs2 = Yintimgs2[conditions2]
    

    forwards1 = Sensor(epsilon = 0.01,mean = mean,snr = snr,device = device,function = function_to_s1)
    forwards2 = Sensor(epsilon = 0.02,mean = mean,snr = snr,device = device,function = function_to_s2)
    f1 = forwards1(coords = Xcordalts1,intensities = Yintimgs1)
    f2 = forwards2(coords = Xcordalts2,intensities = Yintimgs2)

    imgs1_shape = [Imgs1.shape[0], Imgs1.shape[1],int(Yintimgs1.shape[0]/(Imgs1.shape[0]*Imgs1.shape[1]))]
    imgs2_shape = [Imgs2.shape[0], Imgs2.shape[1],int(Yintimgs2.shape[0]/(Imgs2.shape[0]*Imgs2.shape[1]))]

    return list((Xcordalts1,Xcordalts2,f1,f2)), list((coordenadas_test, Yint)), list((imggt_shape,imgs1_shape,imgs2_shape))


#################################

#class from data set

class Coordinatesofimagedataset(Dataset):
    

    def __init__(self, datacoords, dataintensities): #inicializacion de variables
        #self.datavalues = datavalues
        #self.test_data= coordinates_processing(path = data_path) #Pasarle la funcion a la imagen en la clase
        self.dataintensities = dataintensities
        self.datacoords = datacoords
        #self.datacoords = (self.datacoords - self.datacoords.min())/(self.datacoords.max() - self.datacoords.min())

 
    def __len__(self): #numero de datos por epoca
        return len(self.datacoords)

    def __getitem__(self, idx): #Asignacion de indices para pasar por los datos
        
        return self.datacoords[idx], self.dataintensities[idx]