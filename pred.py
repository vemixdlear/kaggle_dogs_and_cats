import torch
from PIL import Image
import torchvision.transforms as transforms
from kaggle_dogs_and_cats.models.AlexNet import Net
import torchvision
import numpy
import os

trans=transforms.Compose([transforms.Resize((224,224)),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.206])])

classes=['cat','dog']

test_dir='./data/kaggle_dogs_and_cats/test1'

net=Net()
net.load_state_dict(torch.load('./models/AlexNet.pth',map_location=torch.device('cpu')))


def read_img(path):
    filedir=os.listdir(path)
    filedir.sort(key=lambda x: int(x[:-4]))
    for filename in filedir:
        filename=path+'/'+filename
        im=Image.open(filename)
        im=trans(im)
        im = torch.unsqueeze(im, 0)

        with torch.no_grad():
            output=net(im)
            pred=torch.max(output,dim=1)[1].data.numpy()
        print('img_name: %s   pred: %s'%(filename,classes[int(pred)]))
#
read_img(test_dir)