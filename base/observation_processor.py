import numpy as np
class queue:
    def __init__(self):
        self.q = []

    def clear(self):
        self.q = []
        
    def append(self, ob):
        self.q.append(ob)

    def getObservation(self, window_length, ob, pic=False):
        state = self.q[-window_length : ]
        if len(state) < window_length:
            state.extend([ob] * (window_length - len(state)))

        if pic:
            return np.array(state)
        return np.array(state).ravel()
