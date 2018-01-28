import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, nb_status, nb_actions, hidden1=600, hidden2=300, init_w=1e-3, use_bn=False):
        super(Actor, self).__init__()
        self.use_bn = use_bn
        self.fc1 = nn.Linear(nb_status, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1, affine=False)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2, affine=False)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.selu = nn.SELU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        out = self.fc1(x)
        if self.use_bn: out = self.bn1(out)
        out = self.selu(out)
        out = self.fc2(out)
        if self.use_bn: out = self.bn2(out)
        out = self.selu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out

class Critic(nn.Module):
    def __init__(self, nb_status, nb_actions, hidden1=600, hidden2=300, init_w=1e-3, use_bn=False):
        super(Critic, self).__init__()
        self.use_bn = use_bn
        self.fc1 = nn.Linear(nb_status+nb_actions, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1, affine=False)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2, affine=False)
        self.fc3 = nn.Linear(hidden2, 1)
        self.selu = nn.SELU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):        
        s, a = x
        out = self.fc1(torch.cat([s, a], 1))
        if self.use_bn: out = self.bn1(out)
        out = self.selu(out)
        out = self.fc2(out)
        if self.use_bn: out = self.bn2(out)
        out = self.selu(out)
        out = self.fc3(out)
        return out
