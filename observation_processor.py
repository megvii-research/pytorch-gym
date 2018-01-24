import numpy as np
class queue:
    def __init__(self):
        self.q = []

    def clear(self):
        self.q = []
        
    def append(self, ob):
        self.q.append(ob)

    def getObservation(self, window_length, ob):
        state = []
        for i in range(window_length):
            if(i < len(self.q)):
                state = np.concatenate((self.q[len(self.q) - i - 1], state))
            else :
                state = np.concatenate((ob, state))
        return np.array(state).ravel()
