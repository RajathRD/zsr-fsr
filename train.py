import os
import torch
import json

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
from datasets.cifar100 import *
from models.resnet import resnet18

# tqdm progressbarfrom:
# https://towardsdatascience.com/training-models-with-a-progress-a-bar-2b664de3e13e
from tqdm import tqdm
from time import sleep

model_dir = "./saved_model"
model_name = "model.pth"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

config = json.load(open("config.json"))
lr = config['lr']
epochs = config['epochs']
batch_size = config['batch_size']
data_dir = config['data_dir']
resume = config['resume']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print (f"Device: {device}")
print('Loading Data...')

train_transforms = [
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
]

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
model = resnet18()

model = model.to(device)

if device == "cuda":
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
#TODO: implement resume train

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=config['lr'],
                      momentum=0.9, weight_decay=0.0005)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def save_model(accuracy):
    print ("Saving Model...")
    torch.save(
        {
            "model": model.state_dict(),
            "accuracy": accuracy 
        },
        os.path.join(model_dir, model_name)
    )

# Training
def train(epoch):
    model.train()
    correct = 0
    total = 0
    with tqdm(train_loader, unit='batch', bar_format="{l_bar}{bar}{n_fmt}/{total_fmt} [{elapsed}] {postfix}") as tepoch:
        for inputs, targets in tepoch:
            tepoch.set_description(f'Epoch: {epoch}')
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
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
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(test_loader, unit='batch') as tepoch:
            for inputs, targets in tepoch:
                tepoch.set_description(f'Epoch: {epoch}')
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                accuracy = 100.*correct/total
                tepoch.set_postfix(
                    loss=loss.item(), 
                    accuracy= accuracy, 
                    correct=correct, 
                    total=total
                )
    
    return accuracy

best_acc = 0
print ("Starting Training...")
for epoch in range(config['epochs']):
    train(epoch)
    test_acc = test(epoch)
    if test_acc > best_acc:
        best_acc = test_acc
        save_model(test_acc)
    scheduler.step()