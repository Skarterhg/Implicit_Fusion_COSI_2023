import torch.nn.functional as F
import torch.nn as nn
import torch
import math

class Sinusoidal(nn.Module):
    def __init__(self, layer_input, layer_output, name = "Sin", w_0=1,device=None):
        super(Sinusoidal, self).__init__()
        self.name = name
        self.device=device
        self.w_0 = w_0
        self.capa = nn.Linear(layer_input, layer_output)

        ## Initialize weights with uniform distribution following torch.nn.init.uniform_
        self.capa.weight.data.uniform_(-math.sqrt(6/(((self.w_0)**2)*layer_input)), math.sqrt(6/(((self.w_0)**2)*layer_input)))
        
    def forward(self, x):
        #print(x.shape)
        x = self.capa(x.to(self.device)*self.w_0)
        return torch.sin(x)

class PE(nn.Module):
    def __init__(self, layer_input, layer_output,meanpe=0,stdpe=1, name = "PE",device=None ):
        super(PE, self).__init__()
        self.name = name
        self.device = device
        self.B=torch.normal(meanpe, stdpe,size=(layer_output,layer_input)).to(self.device)
        self.B=torch.transpose(self.B,1,0)
    def forward(self, x,):
        v=2*torch.pi*torch.matmul(x.to(self.device),self.B.to(self.device)).to(self.device)
        return torch.concatenate([torch.sin(v),torch.cos(v)],dim=-1)

        

class Net(nn.Module):
    def __init__(self, initial_layer_input, layer_output, final_layer_output, num_layers, device = None,w_0=1,meanpe=0,stdpe=1):

        super(Net, self).__init__()
        self.device = device
        self.in_features = initial_layer_input
        self.num_layers = num_layers
        self.net = []
        self.w_0 = w_0
        if stdpe!=-1:
           print("Using PE")
           self.net.append(PE(layer_input = initial_layer_input, layer_output = layer_output,device=self.device,meanpe=meanpe,stdpe=stdpe))
           self.net.append(Sinusoidal(layer_input = 2*layer_output, layer_output = layer_output,device=device))
        #self.net.append(PE(layer_input = initial_layer_input, layer_output = layer_output,device=self.device,meanpe=meanpe,stdpe=stdpe))
        else:
            self.net.append(Sinusoidal(layer_input = initial_layer_input, layer_output = layer_output,device=device))


        for i in range(num_layers-2):
            self.net.append(Sinusoidal(layer_output, layer_output,w_0=w_0))
        
        self.net.append(Sinusoidal(layer_output, final_layer_output,w_0=w_0))

        self.net = nn.ModuleList(self.net)
        
    def get_histogram_of_layers(self):
        x = (torch.rand(2*10, self.in_features)*2 - 1).to(self.device)
        outputs = [x]
        names = ["Input"]
        if len(x.shape) == 1:
            x = x.reshape(1, x.shape[0])
        for i, l in enumerate(self.net):
            x = l(x)
            outputs.append(x)
            names.append(l.name)
        return names, outputs   

    def forward(self, x):
       # x = x.view(-1, 2)
    # for i in range(self.num_layers):
    
    #       x = self.net[i](x)
        for layer in self.net:

            #print("X shape",x.shape)
            x = layer(x)

        return x
    


