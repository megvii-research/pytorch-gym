import numpy as np


class Myrandom():
    def __init__(self, size):
        self.size = size

    def sample(self):
        return np.random.normal(size=self.size)

    def reset_status(self):
        pass
