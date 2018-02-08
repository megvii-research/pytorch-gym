from torch import nn
from gym.spaces import Box, Discrete
from models.policies import DiagGauss, Categorical, StochPolicy
from models.baselines import ValueFunction
from basic_utils.utils import *
from basic_utils.layers import *


class MLPs_pol(nn.Module):
    def __init__(self, ob_space, net_topology, output_layers):
        super(MLPs_pol, self).__init__()
        self.layers = nn.ModuleList([])
        inshp = ob_space.shape[0]
        for des in net_topology:
            l, inshp = get_layer(des, inshp)
            self.layers.append(l)
        self.layers += output_layers(inshp)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class MLPs_v(nn.Module):
    def __init__(self, ob_space, net_topology):
        super(MLPs_v, self).__init__()
        self.layers = nn.ModuleList([])
        inshp = ob_space.shape[0]
        for (i, des) in enumerate(net_topology):
            l, inshp = get_layer(des, inshp)
            self.layers.append(l)
        self.layers.append(nn.Linear(inshp, 1))

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x
