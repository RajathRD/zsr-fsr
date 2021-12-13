import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from models.resnet import *

class OneShotModel(nn.Module):
    def __init__(self, num_outputs=100):
        super(OneShotModel, self).__init__()

        self.resnet = resnet18(num_outputs)
    
    def forward(self, img, target_img):
        out_img = self.resnet(img)
        out_target = self.resnet(target_img)

        return out_img, out_target