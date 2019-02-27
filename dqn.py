import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

from settings import s

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        k1,k2,k3 = 3,3,4
        s1,s2,s3 = 1,1,2
        h = s[9]-2
        self.conv1 = nn.Conv2d(4, 8, kernel_size=k1, stride=s1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=k2, stride=s2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=k3, stride=s3)
        self.bn3 = nn.BatchNorm2d(16)

        dim_conv = int( ((((h-(k1-1))/s1) - (k2-1))/s2 - (k3-1))/s3 )
        linear_input_size = dim_conv * dim_conv * 16
        self.fc1 = nn.Linear(linear_input_size, 60)
        self.fc2 = nn.Linear(60, 6)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

