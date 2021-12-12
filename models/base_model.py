import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from models.resnet import *

class BaseModel(nn.Module):
    def __init__(self, num_outputs=100):
        super(BaseModel, self).__init__()

        self.resnet = resnet18(num_outputs)
    
    def forward(self, x):
        out = self.resnet(x)

        return out