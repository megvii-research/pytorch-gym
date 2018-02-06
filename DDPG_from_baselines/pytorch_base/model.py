import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, nb_status, nb_actions, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_status, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, nb_actions)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.tanh(self.fc3(out))

        return out


class Critic(nn.Module):
    def __init__(self, nb_status, nb_actions, init_w=3e-4):
        super(Critic, self).__init__()
        self.s_fc1 = nn.Linear(nb_status, 64)
        self.fc1 = nn.Linear(64 + nb_actions, 64)
        self.fc2 = nn.Linear(64, 1)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc2.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        s, a = x
        if len(s.size()) == 1:
            s = s.view(1, -1)
        if len(a.size()) == 1:
            a = a.view(1, -1)
        out = F.relu(self.s_fc1(s))
        out = F.relu(self.fc1(torch.cat([out, a], 1)))
        out = self.fc2(out)

        return out
