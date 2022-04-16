import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self,in_channel,out_channel,stride=1,downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1=nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=stride,padding=1)
        self.bn1=nn.BatchNorm2d(out_channel)
        self.relu=nn.ReLU()
        self.conv2=nn.Conv2d(out_channel,out_channel,3,stride,1)
        self.bn2=nn.BatchNorm2d(out_channel)
        self.downsample=downsample

    def forward(self,x):
        identity=x
        if self.downsample is not None:
            identity=self.downsample(x)

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)

        out+=identity
        out=self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self,BasicBlock,num_classes=2,):
        super(ResNet, self).__init__()
        self.in_channel=64
        self.conv1=nn.Conv2d(3,64,7,2,3)
        self.bn1=nn.BatchNorm2d(64)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1=self._make_layer(BasicBlock,64,2,stride=1)
        self.layer2=self._make_layer(BasicBlock,128,2,stride=2)
        self.layer3=self._make_layer(BasicBlock,256,2,stride=2)
        self.layer4=self._make_layer(BasicBlock,512,2,stride=2)

        self.fc=nn.Linear(512,num_classes)




    def _make_layer(self,BasicBlock,channel,block_num,stride):
        strides = [stride] + [1] * (block_num - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channel, channel,stride))
            self.in_channel = channel
        return nn.Sequential(*layers)




