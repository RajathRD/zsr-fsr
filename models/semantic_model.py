import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from models.resnet import *

class SemanticEmbeddingModel(nn.Module):
    def __init__(self, num_outputs):
        super(SemanticEmbeddingModel, self).__init__()

        self.resnet = resnet18(num_outputs)
        self.pre_bn = nn.BatchNorm2d()
        self.linear1 = nn.Linear(300, 512)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d()
        # self.linear2 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 300)
        self.bn2 = nn.BatchNorm2d()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, w):
        out_x = self.relu(self.pre_bn(self.resnet(x)))
        
        out_w = self.linear1(w)
        out_w = self.bn1(out_w)
        out_w = self.relu(out_w)

        out_w = self.linear2(out_w)
        out_w = self.bn2(out_w)
        out_w = self.relu(out_w)
        
        return out_x, out_w