from dataloader import coordinates_processing,Coordinatesofimagedataset,DataLoader
from model import Net
from train import TrainingLoop
from callbacks import LogTrainingCallback
import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from forwardmodel import Sensor, bandselector
from losses import mse, psnr
from numpy import save, savez
from sensingmod import wavelentgh2cords
import cv2 as cv
import argparse
import torch.nn

parser = argparse.ArgumentParser()
parser.add_argument("--cover", type=float, default=-0.0, help="cover of the experiment")
config = parser.parse_args()

imagen_cargar = "pavia"

if imagen_cargar == "pavia":
    mat = scipy.io.loadmat(os.path.join('Train Images','PaviaU.mat'))
    I=mat['paviaU'][..., 1:].astype("float32")
   
    
elif imagen_cargar == "indian_pines":
    mat = scipy.io.loadmat(os.path.join('Train Images','Indian_pines_corrected.mat'))
    I=mat['indian_pines_corrected'][..., 10:].astype("float32")
    
else:
    raise ValueError("No se ha seleccionado una imagen valida")



I = (I- I.min())/(I.max()-I.min())
I =cv.resize(I,(128,128))

data_path = I
size=(data_path.shape[0],data_path.shape[1])
#data_path = os.path.join("Train Images", "gato.jpeg")
debug = False

if debug:
    results_path = "debug"
else:
    results_path =  os.path.join("new_range_results_pavia_lowerpe", "cover_results","optim_parameters")

#BATCH_SIZE = 2**14
#if debug:
 #   BATCH_SIZE = 2**10

SHUFFLE = False
nro_bandas=I.shape[-1]
channels=3
layer_output=512 
num_layers=21 
w_0=30
learning_rate = 1e-6#optimalvalue
stdpe=0.7#1.8207963267948966

n_epochs = 120
steps_per_epoch = None
initial_layer_input = 3
final_layer_output = 1

mean=0
snr = -1
function_to_s1 = lambda x: x+1
function_to_s2 = lambda x: -x+1

start=350
end=650
numberofbands=nro_bandas

lamb1 = 400
lamb2 = 2500
porcentaje = config.cover
wvcut = (lamb2-lamb1)/2*porcentaje
middle = 800#(lamb2+lamb1)/2

lambda0 = wavelentgh2cords(wv = lamb1, lambda0 = lamb1, lambda1 = lamb2)
lambda1 = wavelentgh2cords(wv = middle+wvcut, lambda0 = lamb1, lambda1 = lamb2)
lambda2 = wavelentgh2cords(wv = middle-wvcut, lambda0 = lamb1, lambda1 = lamb2)
lambda3 = wavelentgh2cords(wv = lamb2, lambda0 = lamb1, lambda1 = lamb2)



if debug:
    steps_per_epoch = 10
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

train_data, test_data, img_shapes = coordinates_processing(path = data_path,s1init=lambda0,s1end=lambda1,s2init=lambda2,s2end=lambda3,size=size,debug = debug,bands=nro_bandas,device=device,mean=mean,snr=snr,function_to_s1=function_to_s1,function_to_s2=function_to_s2)
train_datasets1 = Coordinatesofimagedataset(train_data[0],train_data[2])
train_datasets2 = Coordinatesofimagedataset(train_data[1],train_data[3])
test_dataset = Coordinatesofimagedataset(test_data[0],test_data[1])

imshapegt = img_shapes[0]
BATCH_SIZE = imshapegt[0]*nro_bandas

train_loaders1 = DataLoader(train_datasets1, batch_size=BATCH_SIZE, shuffle=True, drop_last = False)
train_loaders2 = DataLoader(train_datasets2, batch_size=BATCH_SIZE, shuffle=True, drop_last = False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last = False)


def forward_function(intensities,coords, sensor):
    if sensor == 1:
        return forwards1(intensities=intensities,coords=coords)
    else:
        return forwards2(intensities=intensities,coords=coords)

forwards1=Sensor(epsilon=0.01,mean=mean,snr=snr,device=device,function=function_to_s1)
forwards2=Sensor(epsilon=0.02,mean=mean,snr=snr,device=device,function=function_to_s2)


model = Net(initial_layer_input=initial_layer_input,layer_output=layer_output,final_layer_output=final_layer_output,num_layers=num_layers,w_0=w_0,device=device,stdpe=stdpe).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = {"MSE":nn.MSELoss()}
evaluation = {"MSE":mse,"PSNR":psnr}

callbacks = LogTrainingCallback(full_dataset=test_loader, exp_folder = results_path+ f"solape%={porcentaje}",eval_loss=evaluation ,shape_data = imshapegt, log_scalars_freq = 1, log_images_freq = 1, write_images = True, write_scalars = True, device = device,start=start,end=end,number=numberofbands)


model_trainer=TrainingLoop(model=model,loss_fn=criterion,optimizer=optimizer,train_loader=(train_loaders1,train_loaders2),model_regularizations=[],losses_weights=[1],regularization_weights=[],full_dataset=None,schedulers=[],callbacks=[callbacks],device=device,forward=forward_function)
model_fit=model_trainer.fit(n_epochs=n_epochs,freq=1,steps_per_epoch=steps_per_epoch)

