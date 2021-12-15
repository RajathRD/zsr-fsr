import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from models.resnet import *

class OneShotBaseModel(nn.Module):
    def __init__(self, num_outputs=100):
        super(OneShotModel, self).__init__()

        self.resnet = resnet18(300)
        self.pre_bn = nn.BatchNorm1d(300)
        self.linear_1 = nn.Linear(300, num_outputs)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, img, target_img):
        embedding = self.resnet(img)
        out = self.linear_1(embedding)

        return out, embedding

class OneShotEmbeddingModel(nn.Module):
    def __init__(self, num_outputs=300):
        super(OneShotModel, self).__init__()

        self.resnet = resnet18(300)
        # self.pre_bn = nn.BatchNorm1d(300)
        # self.linear_1 = nn.Linear(300, num_outputs)
        # self.relu = nn.ReLU(inplace=True)
    
    def forward(self, img, target_img):
        embedding = self.resnet(img)
        
        return embedding