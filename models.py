import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=8, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 10)

        if args.activation == 'relu':
            self.act = nn.ReLU()
        elif args.activation == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError('Unknown activation function: {}'.format(args.activation))

    def forward(self, x):
        x = self.pool1(self.act(self.conv1(x)))
        x = self.pool2(self.act(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if args.activation == 'relu':
            self.act = nn.ReLU()
        elif args.activation == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError('Unknown activation function: {}'.format(args.activation))

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.pool1(self.act(self.conv2(x)))

        x = self.act(self.conv3(x))
        x = self.pool2(self.act(self.conv4(x)))

        x = self.act(self.conv5(x))
        x = self.pool3(self.act(self.conv6(x)))

        x = self.act(self.conv7(x))
        x = self.avgpool(self.conv8(x))
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=1)