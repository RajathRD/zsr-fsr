import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from resnet18 import *

class Devise(nn.Module):
    def __init__(self):
        super().__init__(self)

        self.resnet = resnet18(num_classes=300)

    def forward(self, x):
        return self.resnet(x)


class SemanticProjectionModel(nn.Module):
    def __init__(self):
        super().__init__(self)

        self.resnet = resnet18()
        self.linear1 = nn.Linear(300, 512)
        # self.linear2 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 300)
        for m in self.modules():
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, w):
        out_x = self.resnet(x)
        
        out_w = self.linear1(w)
        out_w = self.linear2(out_w)
        
        return out_x, out_w

