import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from models.resnet import *

class DeviseModel(nn.Module):
    def __init__(self, num_outputs=300):
        super(DeviseModel, self).__init__()

        self.resnet = resnet18(num_outputs)
    
    def normalize(self, x):
        norm = torch.sum(torch.square(x), axis=1)
        
        return x/torch.sqrt(norm).reshape(-1, 1)

    def forward(self, x):
        out = self.resnet(x)
        out = self.normalize(out)
        return out

class SemanticEmbeddingModel(nn.Module):
    def __init__(self, num_outputs):
        super(SemanticEmbeddingModel, self).__init__()

        self.resnet = resnet18(num_outputs)
        
        self.linear1 = nn.Linear(300, 512)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 300)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, w):
        out_x = self.resnet(x)
        out_w = self.linear1(w)
        out_w = self.bn1(out_w)
        out_w = self.relu(out_w)

        out_w = self.relu(self.linear2(out_w))
        
        return out_x, out_w

    def forward_semantic(self, w):
        out_w = self.linear1(w)
        out_w = self.bn1(out_w)
        out_w = self.relu(out_w)

        out_w = self.relu(self.linear2(out_w))
        # print (out_w.shape)
        return out_w
        