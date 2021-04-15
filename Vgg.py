# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 13:06:16 2021

@author: Admin
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary

VGG_layers = {
    'VGG11' : [64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'VGG13' : [64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'VGG16' : [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
    'VGG19' : [64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M']
#Now Flatten and then 4096X4096X1000 Linear Layers
}

class VGG(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(VGG,self).__init__()
        self.in_channels = in_channels
        self.conv_layer = self.create_conv_layers(VGG_layers['VGG19'])
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,num_classes)
            )
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x
     
    def create_conv_layers(self,architecture):
       
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == int:
                out_channels = x
                
                layers += [nn.Conv2d(in_channels=in_channels,
                                     out_channels = out_channels,
                                     kernel_size=(3,3), stride=(1,1), padding = (1,1))
                           , nn.BatchNorm2d(x), nn.ReLU()]
                in_channels = x
            
            elif x == 'M':
            
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
            
        return nn.Sequential(*layers)

#Testing
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VGG(3,1000).to(device)
x = torch.randn(1,3,224,224).to(device)
print(model(x).shape)
print(summary(model, (3,224,224)))




