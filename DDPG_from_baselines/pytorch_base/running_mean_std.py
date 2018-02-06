import numpy as np


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-2):
        self._sum = 0.0
        self._sumsq = epsilon
        self._count = epsilon

        self.mean = self._sum / self._count
        self.std = np.sqrt(np.max([self._sumsq / self._count - np.square(self.mean), 1e-2]))

    def update(self, x):
        x = x.astype('float64')
        self._sum += x.sum(axis=0)
        self._sumsq += np.square(x).sum(axis=0)
        self._count += len(x)

        self.mean = self._sum / self._count
        self.std = np.sqrt(np.clip(self._sumsq / self._count - np.square(self.mean), 1e-2, np.inf))
