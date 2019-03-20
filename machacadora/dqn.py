import torch
import torch.nn as nn
import torch.nn.functional as F

class twolayer_4input(nn.Module):

    def __init__(self):
        super(twolayer_4input, self).__init__()
        self.decision = nn.Sequential(
            nn.Linear(4, 6, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(6, 6),
        )
        self.name = 'twolayer_4input'

    def forward(self, x):
        out = self.decision(x)
        return out

class onelayer_4input(nn.Module):

    def __init__(self):
        super(onelayer_4input, self).__init__()
        self.decision = nn.Sequential(
            nn.Linear(4, 6, bias=True)
        )
        self.name = 'onelayer_4input'

    def forward(self, x):
        out = self.decision(x)
        return out
    

class onelayer_14input(nn.Module):

    def __init__(self):
        super(onelayer_14input, self).__init__()
        self.decision = nn.Sequential(
            nn.Linear(14, 6, bias=True)
        )
        self.name = 'onelayer_14input'

    def forward(self, x):
        out = self.decision(x)
        return out


class threelayer_4input(nn.Module):

    def __init__(self):
        super(threelayer_4input, self).__init__()
        self.decision = nn.Sequential(
            nn.Linear(4, 6, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(6, 6),
            nn.LeakyReLU(inplace=True),
            nn.Linear(6, 6),
        )
        self.name = 'threelayer_4input'

    def forward(self, x):
        out = self.decision(x)
        return out


class twolayer_14input(nn.Module):

    def __init__(self):
        super(twolayer_14input, self).__init__()
        self.decision = nn.Sequential(
            nn.Linear(14, 6, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(6, 6),
        )
        self.name = 'twolayer_14input'

    def forward(self, x):
        out = self.decision(x)
        return out


class threelayer_14input(nn.Module):

    def __init__(self):
        super(threelayer_14input, self).__init__()
        self.decision = nn.Sequential(
            nn.Linear(14, 6, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(6, 6),
            nn.LeakyReLU(inplace=True),
            nn.Linear(6, 6),
        )
        self.name = 'threelayer_14input'

    def forward(self, x):
        out = self.decision(x)
        return out

