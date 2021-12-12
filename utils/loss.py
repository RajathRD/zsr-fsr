import torch
import torch.nn as nn
from torch.nn.functional import normalize
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DeviseLoss(nn.Module):
    def __init__(self, word_vectors, target_classes, classes):
        super(DeviseLoss, self).__init__()
        self.word_vectors = word_vectors
        self.classes = classes
        self.target_classes = target_classes

    def forward(self, outputs, targets, target_wvs):
        loss = 0
        margin = 0.1
        true_distance = torch.sum(torch.multiply(outputs, target_wvs), axis=1)
        for j in self.target_classes:
            j_distance = torch.matmul(outputs, torch.Tensor(self.word_vectors[self.classes[j]]).to(device))
            l = torch.clamp(margin - true_distance + j_distance, min=0)
            loss += torch.sum(l)
        loss /= outputs.shape[0]
        return loss
