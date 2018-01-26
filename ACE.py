import numpy as np
from util import *
from ddpg import DDPG

class ACE:
    def __init__(self, nb_status, nb_actions, args):
        self.ensemble = []
        self.num = args.ace
        self.iter = 0
        for i in range(self.num):
            self.ensemble.append(DDPG(nb_status, nb_actions, args))
        self.discrete = args.discrete

    def __call__(self, st):
        status = []
        actions = []
        tot_score = []
        for i in range(self.num):
            if i > self.iter: break
            action = self.ensemble[i].select_action(st, return_fix=True)
            actions.append(action)
            status.append(st)
            tot_score.append(0.)

        for i in range(self.num):
            self.ensemble[i].eval()
            if i > self.iter: break
            score = self.ensemble[i].critic([
                to_tensor(np.array(status), volatile=True), to_tensor(np.array(actions), volatile=True)
            ])
            for j in range(self.num):
                if j > self.iter: break
                tot_score[j] += score.data[j][0]
            self.ensemble[i].train()
                
        best = np.array(tot_score).argmax()
        best = 0
        if self.discrete:
            return actions[best].argmax()
        return actions[best]

    def append(self, output, num):
        self.ensemble[self.iter % self.num].load_weights(output, num)
        self.iter += 1

    def load(self, output):
        for i in range(self.num):
            self.ensemble[i].load_weights(output)
