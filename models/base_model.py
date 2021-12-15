import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from models.resnet import *

class CNN(nn.Module):
    def __init__(self, num_outputs):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding='same')
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv2_drop = nn.Dropout2d()

        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding='same')
        self.conv4_bn = nn.BatchNorm2d(32)
        self.conv4_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(8*8*32, 256)
        self.fc1_bn = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, num_outputs)

        self.act = F.relu

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(self.conv1_bn(x))
        x = self.conv2(x)
        x = self.act(F.max_pool2d(self.conv2_drop(self.conv2_bn(x)), 2))
        x = self.conv3(x)
        x = self.act(self.conv3_bn(x))        
        x = self.conv4(x)
    
        x = self.act(F.max_pool2d(self.conv4_drop(self.conv4_bn(x)), 2))
        
        x = x.view(-1, 8*8*32)
        x = self.fc1(x)
        x = self.act(self.fc1_bn(x))

        x = self.fc2(x)
        return self.act(x)

class BaseModel(nn.Module):
    def __init__(self, num_outputs=100):
        super(BaseModel, self).__init__()

        self.resnet = resnet18(num_outputs)
    
    def forward(self, x):
        out = self.resnet(x)

        return out