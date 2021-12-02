import torch
import json

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
from datasets.cifar100 import *
from models.resnet import resnet18

# tqdm progress from:
# https://towardsdatascience.com/training-models-with-a-progress-a-bar-2b664de3e13e
from tqdm import tqdm
from time import sleep

config = json.load(open("config.json"))
lr = config['lr']
epochs = config['epochs']
batch_size = config['batch_size']
data_dir = config['data_dir']
resume = config['resume']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print (f"Device: {device}")
print('Loading Data...')

train_transforms = []

# normalization constants from
# https://www.programcreek.com/python/example/105099/torchvision.datasets.CIFAR100
transforms = [
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
]

train_data, test_data = cifar100Original(
    data_dir, 
    train_transforms,
    transforms
)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=2)


test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False, num_workers=2)

print("Creating Model...")
net = resnet18()

net = net.to(device)

if device == "cuda":
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
#TODO: implement resume train

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=config['lr'],
                      momentum=0.9, weight_decay=0.0005)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    with tqdm(train_loader, unit='batch') as tepoch:
        for inputs, targets in tepoch:
            tepoch.set_description(f'Epoch: {epoch}')
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            tepoch.set_postfix(
                loss=loss.item(), 
                accuracy=100.*correct/total, 
                correct=correct, 
                total=total
            )
            # sleep(0.1)

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(test_loader, unit='batch') as tepoch:
            for inputs, targets in tepoch:
                tepoch.set_description(f'Epoch: {epoch}')
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                tepoch.set_postfix(
                    loss=loss.item(), 
                    accuracy=100.*correct/total, 
                    correct=correct, 
                    total=total
                )
                # sleep(0.1)

print ("Starting Training...")
for epoch in range(config['epochs']):
    train(epoch)
    test(epoch)
    scheduler.step()