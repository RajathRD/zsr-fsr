import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.backends.cudnn as cudnn

from sklearn.neighbors import KNeighborsClassifier
from datasets.dataloader import *
from models.base_model import *
from sklearn.metrics import top_k_accuracy_score
from utils.loss import *
# tqdm progressbarfrom:
# https://towardsdatascience.com/training-models-with-a-progress-a-bar-2b664de3e13e
from tqdm import tqdm
from time import sleep

config_file = "./configs/config_wv.json"

config = json.load(open(config_file))
print (f"Loaded: {config_file}")
print(config)
model_dir = config['model_dir']
model_name = config['model_name']
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

vector_type = config['vector_type']
dataset = config['dataset']
lr = config['lr']
epochs = config['epochs']
batch_size = config['batch_size']
data_dir = config['data_dir']
resume = config['resume']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print (f"Device: {device}")
print(f'Loading Data ({data_dir, dataset})...')
print(f'Models will be saved in {model_dir} as {model_name}')

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

if dataset == "cifar_original":
    train_data, test_data = cifarOriginal(
        data_dir, 
        train_transforms,
        transforms
    )
elif dataset == "cifar_zs":
    train_data = cifarZSClassification(data_dir, train_transforms, transforms, train=True)
    test_data = cifarZSClassification(data_dir, train_transforms, transforms, train=False)
    print (f"Train Size: {len(train_data.targets)} Test Size: {len(test_data.targets)}\
            \nTrain Classes: {len(train_data.target_classes)} Test Classes: {len(test_data.target_classes)}")

elif dataset == "cifar_wv":
    
    train_data = cifarZSW2V(data_dir, train_transforms, transforms, train=True, vector_type=vector_type)
    test_data = cifarZSW2V(data_dir, train_transforms, transforms, train=False, vector_type=vector_type)

elif dataset == 'cifar_oneshot':
    train_data = cifarKShot(data_dir, train_transforms, transforms, train=True, k=1)
    test_data = cifarKShot(data_dir, train_transforms, transforms, train=False, k=1)
    

train_classifier = KNeighborsClassifier(n_neighbors=1)
test_classifier = KNeighborsClassifier(n_neighbors=1)

train_classifier.fit(train_data.target_wv, np.arange(len(train_data.target_classes)))
test_classifier.fit(test_data.target_wv, np.arange(len(test_data.target_classes)))

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False, num_workers=2)

print("Creating Model...")
model = BaseModel(train_data.word_vectors['apple'].shape[0])
# model = CNN(train_data.word_vectors['apple'].shape[0])
criterion = nn.MSELoss()
# criterion = TripletLoss()
model = model.to(device)

if device == "cuda":
    cudnn.benchmark = True
#TODO: implement resume train

optimizer = optim.SGD(model.parameters(), lr=config['lr'],
                      momentum=0.9, weight_decay=0.0005)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def save_model(epoch, accuracy, meta):
    print ("Saving Model...")
    state_dict = {
            "epoch": epoch,
            "model": model.state_dict(),
            "accuracy": accuracy 
        }
    state_dict.update(meta)
    torch.save(
        state_dict,
        os.path.join(model_dir, model_name)
    )

# Training
def train(epoch):
    model.train()
    correct = 0
    total = 0
    all_probas = []
    all_targets = []
    with tqdm(train_loader, unit='batch', bar_format="{l_bar}{bar}{n_fmt}/{total_fmt} [{elapsed}] {postfix}") as tepoch:
        for batch_idx, (inputs, target_wvs, targets, negatives) in enumerate(tepoch):
            tepoch.set_description(f'Epoch: {epoch}')
            inputs, target_wvs = inputs.to(device), target_wvs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, target_wvs)
            loss.backward()
            optimizer.step()
            predicted = train_classifier.predict(outputs.cpu().data.numpy())
            preds = torch.Tensor(train_data.target_classes[predicted])
            predict_probas = train_classifier.predict_proba(outputs.cpu().data.numpy())
            all_targets += targets.cpu().data.tolist()
            all_probas += predict_probas.tolist()

            total += targets.size(0)
            correct += (preds == targets).sum().item()
            accuracy = 100.*correct/total
            tepoch.set_postfix(
                loss=loss.item(), 
                accuracy=accuracy, 
                correct=correct, 
                total=total
            )

def test(epoch):
    model.eval()

    correct = 0
    total = 0
    all_probas = []
    all_targets = []
    with torch.no_grad():
        with tqdm(test_loader, unit='batch') as tepoch:
            for inputs, target_wvs, targets, negatives in tepoch:
                tepoch.set_description(f'Epoch: {epoch}')
                inputs, target_wvs = inputs.to(device), target_wvs.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, target_wvs)
                predicted = test_classifier.predict(outputs.cpu().data.numpy())
                predict_proba = test_classifier.predict(outputs.cpu().data.numpy())
                preds = torch.Tensor(test_data.target_classes[predicted])
                predict_probas = test_classifier.predict_proba(outputs.cpu().data.numpy())
                all_targets += targets.cpu().data.tolist()
                all_probas += predict_probas.tolist()

                total += targets.size(0)
                correct += (preds == targets).sum().item()
                accuracy = 100.*correct/total
                tepoch.set_postfix(
                    loss=loss.item(), 
                    accuracy=accuracy, 
                    correct=correct, 
                    total=total
                )

    accuracy_at_k = (top_k_accuracy_score(all_targets, all_probas, k=3)*100)
    print ("Accuracy@3:", accuracy_at_k)
    return accuracy, accuracy_at_k

best_acc = 0
print ("Starting Training...")
for epoch in range(config['epochs']):
    train(epoch)
    test_acc, top_k_acc = test(epoch)
    if test_acc > best_acc:
        best_acc = test_acc
        save_model(epoch, test_acc, {"top_k": top_k_acc})
    scheduler.step()