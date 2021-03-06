import os
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision
import json
import numpy as np
import pickle 
import torch

from PIL import Image
from torch.utils.data import Dataset

config = json.load(open("./configs/config_base.json"))

def cifarOriginal(data_dir, train_transforms, transforms):
    train_data = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=T.Compose(train_transforms+transforms))

    test_data = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=T.Compose(transforms))

    return train_data, test_data

class cifarZSClassification(Dataset):
    def __init__(self, data_dir, train_transforms, transforms, train):
        self.train = train
        if self.train == True:
            self.transform = T.Compose(train_transforms + transforms)
            torch_data = torchvision.datasets.CIFAR100(
                root=data_dir, train=True, download=True, transform=self.transform)    
        else:
            self.transform = T.Compose(transforms)
            torch_data = torchvision.datasets.CIFAR100(
                root=data_dir, train=False, download=True, transform=self.transform)

        test_classes = [torch_data.classes.index(c) for c in config["test_classes"]]

        if train == True:
            self.target_classes = np.delete(np.arange(100), test_classes)
        else:
            self.target_classes = np.array(test_classes)

        self.indices = [i for i in range(len(torch_data.targets)) if torch_data.targets[i] in self.target_classes]

        self.data = torch_data.data[self.indices]
        self.targets = list(np.array(torch_data.targets)[self.indices])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

class cifarZSW2V(Dataset):
    def __init__(self, data_dir, train_transforms, transforms, train, vector_type='glove'):
        self.train = train
        np.random.seed(10)
        if train == True:
            self.transform = T.Compose(train_transforms + transforms)
            torch_data = torchvision.datasets.CIFAR100(
                root=data_dir, train=True, download=True, transform=self.transform)    
        else:
            self.transform = T.Compose(transforms)
            torch_data = torchvision.datasets.CIFAR100(
                root=data_dir, train=False, download=True, transform=self.transform)

        test_classes = [torch_data.classes.index(c) for c in config["test_classes"]]
        self.classes = torch_data.classes
        self.word_vectors = pickle.load(open(os.path.join(config['w2v_dir'], config[vector_type]), 'rb'))
        # print (self.word_vectors)
        if train == True:
            self.target_classes = np.delete(np.arange(100), test_classes)
        else:
            self.target_classes = np.array(test_classes)
        # print (self.classes, len(self.classes))
        # print ()
        self.target_wv = [self.word_vectors[self.classes[idx]] for idx in self.target_classes]
        
        self.indices = [i for i in range(len(torch_data.targets)) if torch_data.targets[i] in self.target_classes]

        self.data = torch_data.data[self.indices]
        self.targets = list(np.array(torch_data.targets)[self.indices])
        self.negatives = [self.word_vectors[self.classes[np.random.choice([c for c in self.target_classes if c != self.targets[idx]])]] for idx in range(len(self.data))]

    def __getitem__(self, index):
        img, target = self.data[index], self.word_vectors[self.classes[self.targets[index]]]
        target_class = self.targets[index]
        negative = self.negatives[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        # print (target)
        target = TF.Tensor(target) 
        negative = TF.Tensor(negative)
        return img, target, target_class, negative

    def __len__(self):
        return len(self.data)

class cifarKShot(Dataset):
    def __init__(self, data_dir, train_transforms, transforms, train, k=1):
        self.k = k
        self.train = train
        if train == True:
            self.transform = T.Compose(train_transforms + transforms)
            torch_data = torchvision.datasets.CIFAR100(
                root=data_dir, train=True, download=True, transform=self.transform)    
        else:
            self.transform = T.Compose(transforms)
            torch_data = torchvision.datasets.CIFAR100(
                root=data_dir, train=False, download=True, transform=self.transform)

        test_classes = [torch_data.classes.index(c) for c in config["test_classes"]]
        support = pickle.load(open(os.path.join(config['data_dir'], config['support_file']), "rb"))
        self.support_data, self.support_indices = support["images"], support["indices"]
        if train == True:
            self.target_classes = np.delete(np.arange(100), test_classes)
        else:
            self.target_classes = np.array(test_classes)

        self.indices = [i for i in range(len(torch_data.targets)) if torch_data.targets[i] in self.target_classes]

        self.data = torch_data.data[self.indices]
        self.targets = list(np.array(torch_data.targets)[self.indices])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if not self.train:
            query = self.support_data[self.targets[index]][0]
            query = Image.fromarray(query)
        
        if self.transform is not None:
            img = self.transform(img)
            if not self.train:
                query = self.transform(query)
        
        if self.train:
            return img, target
        else:
            return img, query, target

    def __len__(self):
        return len(self.data)

class kShotSupport(Dataset):
    def __init__(self, data_dir, transforms, train, k=1):
        self.k = k
        self.train = train
        if train == True:
            self.transform = T.Compose(transforms)
            torch_data = torchvision.datasets.CIFAR100(
                root=data_dir, train=True, download=True, transform=self.transform)    
        else:
            self.transform = T.Compose(transforms)
            torch_data = torchvision.datasets.CIFAR100(
                root=data_dir, train=False, download=True, transform=self.transform)

        test_classes = [torch_data.classes.index(c) for c in config["test_classes"]]
        support = pickle.load(open(os.path.join(config['data_dir'], config['support_file']), "rb"))
        self.support_data, self.support_indices = support["images"], support["indices"]
        if train == True:
            self.target_classes = np.delete(np.arange(100), test_classes)
        else:
            self.target_classes = np.array(test_classes)

        self.indices = [i for i in range(len(torch_data.targets)) if torch_data.targets[i] in self.target_classes]
        # self.targets = list(np.array(torch_data.targets)[self.indices])

        self.data = []
        self.targets = []
        for c_i in self.target_classes:
            for j in range(self.k):
                img = self.support_data[c_i][0]
                img = Image.fromarray(img)
                img = self.transform(img)
                self.data.append(img)
                self.targets.append(c_i)

        
        self.data = torch.stack(self.data)
        print ("Support Data: ", self.data.shape, self.targets)
        
        
