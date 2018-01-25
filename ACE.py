import numpy as np
from util import *
from ddpg import DDPG

class ACE:
    def __init__(self, nb_status, nb_actions, args):
        self.ensemble = []
        self.num = 5
        self.iter = 0
        for i in range(self.num):
            self.ensemble.append(DDPG(nb_status, nb_actions, args))
        self.discrete = args.discrete

    def __call__(self, st):
        actions = []
        tot_score = []
        for i in range(self.num):
            if i > self.iter: break
            action = self.ensemble[i].select_action(st, return_fix=True)
            actions.append(action)
            tot_score.append(0.)
            for j in range(self.num):
                if j > self.iter: break
                score = self.ensemble[j].critic([
                    to_tensor(np.array([st]), volatile=True), to_tensor(np.array([action]), volatile=True)
                ])
                tot_score[i] += score.data[0][0]
        best = np.array(tot_score).argmax()
        if self.discrete:
            return actions[best].argmax()
        return actions[best]

    def append(self, output, num):
        self.ensemble[self.iter % self.num].load_weights(output, num)
        self.iter += 1

    def load(self, output):
        for i in range(self.num):
            self.ensemble[i].load_weights(output)
