import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def fanin_init(size, fanin=None, init_method='uniform'):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    if init_method == 'uniform':
        return torch.Tensor(size).uniform_(-v, v)
    else:
        return torch.Tensor(size).normal_(-v, v)

class Actor(nn.Module):
    def __init__(self, nb_status, nb_actions, hidden1=400, hidden2=300, init_w=3e-3, use_bn=False, use_bn_affine=False, init_method='uniform'):
        super(Actor, self).__init__()
        self.use_bn = use_bn or use_bn_affine
        self.bn1 = nn.BatchNorm1d(hidden1, affine=use_bn_affine)
        self.fc1 = nn.Linear(nb_status, hidden1)
        self.bn2 = nn.BatchNorm1d(hidden1, affine=use_bn_affine)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn3 = nn.BatchNorm1d(hidden2, affine=use_bn_affine)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.selu = nn.SELU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w, init_method)
    
    def init_weights(self, init_w, init_method):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size(), init_method=init_method)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size(), init_method=init_method)
        if init_method == 'uniform':
            self.fc3.weight.data.uniform_(-init_w, init_w)
        else:
            self.fc3.weight.data.normal_(-init_w, init_w)
    
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
    def __init__(self, nb_status, nb_actions, hidden1=400, hidden2=300, init_w=3e-4, use_bn=False, use_bn_affine=False, init_method='uniform'):
        super(Critic, self).__init__()
        self.use_bn = use_bn or use_bn_affine
        self.fcs = nn.Linear(nb_status, hidden1 // 2)
        self.fca = nn.Linear(nb_actions, hidden1 // 2)
        self.bns = nn.BatchNorm1d(hidden1 // 2, affine=use_bn_affine)
        self.bna = nn.BatchNorm1d(hidden1 // 2, affine=use_bn_affine)
        self.fc1 = nn.Linear(hidden1, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1, affine=use_bn_affine)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2, affine=use_bn_affine)
        self.fc3 = nn.Linear(hidden2, 1)
        self.selu = nn.SELU()
        self.init_weights(init_w, init_method)

    def init_weights(self, init_w, init_method):
        self.fcs.weight.data = fanin_init(self.fcs.weight.data.size(), init_method=init_method)
        self.fca.weight.data = fanin_init(self.fca.weight.data.size(), init_method=init_method)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size(), init_method=init_method)
        if init_method == 'uniform':
            self.fc3.weight.data.uniform_(-init_w, init_w)
        else:
            self.fc3.weight.data.normal_(0, init_w)

    def forward(self, x):
        s, a = x
        s = self.fcs(s)
        if self.use_bn: s = self.bns(s)
        s = self.selu(s)
        a = self.fca(a)
        if self.use_bn: a = self.bns(a)
        a = self.selu(a)
        
        out = self.fc1(torch.cat([s, a], 1))
        if self.use_bn: out = self.bn1(out)
        out = self.selu(out)
        
        out = self.fc2(out)
        if self.use_bn: out = self.bn2(out)
        out = self.selu(out)

        out = self.fc3(out)
        return out
