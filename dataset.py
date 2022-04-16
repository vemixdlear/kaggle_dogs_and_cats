import os
import torch
import torchvision
import torchvision.datasets as datasets
import shutil
from torch.utils.data import DataLoader


base_dir= './data/kaggle_dogs_and_cats/train'

test_dir= './data/kaggle_dogs_and_cats/test1'

train_dir = './data/kaggle_dogs_and_cats/train2'
val_dir='./data/kaggle_dogs_and_cats/val'

if not os.path.isdir(train_dir):
    os.mkdir(train_dir)

if not os.path.isdir(val_dir):
    os.mkdir(val_dir)

classes=['cat','dog']
for clas in classes:
    if not os.path.isdir(os.path.join(train_dir, clas)):
        os.mkdir(os.path.join(train_dir,clas))

for clas in classes:
    if not os.path.isdir(os.path.join(val_dir, clas)):
        os.mkdir(os.path.join(val_dir,clas))

#训练集
fnames = ['cat.{}.jpg'.format(i) for i in range(10000)]
for fname in fnames:
    s = os.path.join(base_dir,fname)
    d = os.path.join(train_dir,'cat',fname)
    shutil.copyfile(s,d)

fnames = ['dog.{}.jpg'.format(i) for i in range(10000)]
for fname in fnames:
    s = os.path.join(base_dir,fname)
    d = os.path.join(train_dir,'dog',fname)
    shutil.copyfile(s,d)

#验证集
fnames = ['cat.{}.jpg'.format(i) for i in range(10000,12500)]
for fname in fnames:
    s = os.path.join(base_dir,fname)
    d = os.path.join(val_dir,'cat',fname)
    shutil.copyfile(s,d)
#
#
fnames = ['dog.{}.jpg'.format(i) for i in range(10000,12500)]
for fname in fnames:
    s = os.path.join(base_dir,fname)
    d = os.path.join(val_dir,'dog',fname)
    shutil.copyfile(s,d)


