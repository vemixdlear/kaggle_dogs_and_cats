import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision
import os
import shutil
import torch.nn.functional as F
from kaggle_dogs_and_cats.models.AlexNet import Net

#训练集 验证集 测试集 划分



train_dir='./data/kaggle_dogs_and_cats/train2'
val_dir='./data/kaggle_dogs_and_cats/val'
test_dir='./data/kaggle_dogs_and_cats/test1'

batch_size=8

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.206])
])


val_transform= torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.206])
])

train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=0)

val_data=datasets.ImageFolder(root=val_dir,transform=val_transform)
val_loader=DataLoader(val_data,batch_size=batch_size,shuffle=True,num_workers=0)


# test_data = datasets.ImageFolder(root='./datasets/kaggle_dogs_and_cats/test1', transform=data_transform)
# test_loader = DataLoader(test_data, batch_size=4, shuffle=True)
#
#

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
loss = nn.CrossEntropyLoss()
optims = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

epochs = 10
# print(net)



def train():
    for epoch in range(epochs):
        train_correct = 0
        running_loss = 0
        for i, data in enumerate(train_loader, 0):
            img, label = data
            img=img.to(device)
            label=label.to(device)
            optims.zero_grad()
            output=net(img)
            result_loss = loss(output, label)
            result_loss.backward()
            optims.step()
            running_loss += result_loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        # 模型测试
    with torch.no_grad():
        net.eval()
        total_acc=0
        for data in val_loader:
            imgs,target=data
            imgs=imgs.to(device)
            target=target.to(device)
            outputs = net(imgs)
            predicted = torch.max(outputs, 1)[1]
            print(target,predicted)
            accuracy = (predicted == target).sum().item()/target.size(0)
            total_acc+=accuracy
        print('test_accuracy: %.3f' % (total_acc / len(val_loader)))
    print('finish train')


train()

safe_path='./models/AlexNet.pth'
torch.save(net.state_dict(),safe_path)
