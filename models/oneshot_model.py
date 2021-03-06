import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from models.resnet import *

class OneShotBaseModel(nn.Module):
    def __init__(self, num_outputs=100):
        super(OneShotBaseModel, self).__init__()

        self.resnet = resnet18(300)
        self.pre_bn = nn.BatchNorm1d(300)
        self.linear_1 = nn.Linear(300, num_outputs)
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                nn.init.constant_(m.bias, 0)
    def forward(self, img):
        embedding = self.resnet(img)
        out = self.relu(self.linear_1(embedding))

        return out, embedding

class Siamese(nn.Module):
    def __init__(self, num_outputs=300):
        super(Siamese, self).__init__()

        self.resnet = resnet18(300)
        # self.pre_bn = nn.BatchNorm1d(300)
        # self.linear_1 = nn.Linear(300, num_outputs)
        # self.relu = nn.ReLU(inplace=True)
    
    def forward(self, img, target):
        embedding = self.resnet(img)
        embedding = self.resnet(target)
        return embedding