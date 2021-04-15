# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 15:37:58 2021

@author: Admin
"""

import torch
import torch.nn as nn

class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels,kernel_size=1, stride = 1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels, kernel_size = 3, stride =stride, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size =1, stride = 1, padding = 0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample  = identity_downsample
        
    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.identity_downsample is not None:
            
            identity = self.identity_downsample(identity)
        
        x += identity
        x = self.relu(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, block, layer, img_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv = nn.Conv2d(img_channels, 64, kernel_size = 7, stride = 2, padding =3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size = (3,3), stride = (2,2), padding = 1)
        
        self.layer1 = self.make_layer(block, layer[0], 64, 1)
        self.layer2 = self.make_layer(block, layer[1], 128, 2)
        self.layer3 = self.make_layer(block, layer[2], 256, 2)
        self.layer4 = self.make_layer(block, layer[3], 512, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)
        
    def forward(self, x):
        
        x = self.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
    
    def make_layer(self, block, num_res_blocks, out_channels, stride):
        layers = []
        identity_downsample = None
        if stride != 1 or self.in_channels != out_channels*4:

            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride = stride),
                                                nn.BatchNorm2d(out_channels*4))
        
        layers.append(block(self.in_channels,out_channels,identity_downsample, stride))
        self.in_channels = out_channels*4
        
        for i in range(num_res_blocks - 1):
            layers.append(block(self.in_channels,out_channels))
        
        return nn.Sequential(*layers)
    
def ResNet100(img_channels = 3, num_classes=1000):
    return ResNet(block, [3,4,23,3], img_channels,num_classes)

model = ResNet100()
from torchsummary import summary
print(summary(model, (3,224,224), device = 'cpu'))
x = torch.randn(3,3,224,224)
y = model(x)
print(y.shape)
       
        
        
        
        
        
        
    
        
        
    
    
    
    
    
    
        