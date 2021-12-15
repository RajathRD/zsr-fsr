import torch
import torch.nn as nn
from torch.nn.functional import normalize
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DeviseLoss(nn.Module):
    # essentially hinge + triplet across all negatives in vocab
    def __init__(self, word_vectors, target_classes, classes):
        super(DeviseLoss, self).__init__()
        self.word_vectors = word_vectors
        self.classes = classes
        self.target_classes = target_classes

    # cosine
    def distance(self, a, b):
        return torch.sum(torch.multiply(a, b), axis=1)
    
    # euclidean / mse
    # def distance(self, a, b):
    #     return torch.sum(torch.square(a - b), axis=1)

    def forward(self, outputs, target_wvs, targets):
        loss = 0
        margin = 0.2
        true_distance = self.distance(outputs, target_wvs)
        for j in self.target_classes:
            j_wv = torch.Tensor(self.word_vectors[self.classes[j]]).to(device)
            j_distance = self.distance(outputs, j_wv)
            # j_distance =
            l = torch.clamp(margin + true_distance - j_distance, min=0)
            loss += torch.sum(l)
        loss /= outputs.shape[0]
        return loss

class DeviseLossWNegative(nn.Module):
    def __init__(self):
        super(DeviseLossWNegative, self).__init__()
        self.margin = 0.2
    
    # cosine
    # def distance(self, a, b):
    #     return torch.sum(torch.multiply(a, b), axis=1)

    # euclidean / mse
    def distance(self, a, b):
        return torch.sum(torch.square(a - b), axis=1)

    def forward(self, outputs, target_wvs, negatives):
        loss = 0
        
        true_distance = self.distance(outputs, target_wvs)
        neg_distance = self.distance(outputs, negatives)
        loss = true_distance - neg_distance
        loss = torch.sum(loss, axis=0)
        loss /= outputs.shape[0]

        return loss

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
    
    def forward(self, outputs, target_wvs):
        margin = 0.1
        true_distance = torch.sum(torch.multiply(outputs, target_wvs), axis=1)
        true_distance = torch.clamp(margin - true_distance, min=0)
        loss = torch.sum(true_distance)
        loss /= outputs.shape[0]
        
        return loss

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def distance(self, a, b):
        return torch.sum(torch.square(a - b), axis=1)

    def forward(self, outputs, target_wvs, negatives):
        loss = 0        
        true_distance = self.distance(outputs, target_wvs)
        # neg_distance = self.distance(outputs, negatives)
        # loss = torch.clamp(self.margin + true_distance - neg_distance, 0)
        # loss = torch.mean(true_distance)
        loss = torch.sum(true_distance)
        loss /= outputs.shape[0]

        return loss
