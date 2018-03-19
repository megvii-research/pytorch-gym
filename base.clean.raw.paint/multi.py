import numpy as np
# preprocess raw image to 84*84 gray image
def preprocess(observation):
    import cv2
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110,:]
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(84,84,1))

class fastenv():
    def __init__(self, env, action_repeat, vis=False, atari=False):
        self.action_repeat = action_repeat
        self.q = []
        self.env = env
        self.vis = vis
        self.atari = atari
        
    def step(self, action):
        tot_reward = 0.
        for i in range(self.action_repeat):
            observation, reward, done, info = self.env.step(action)
            tot_reward += reward
            if done:
                break
            if self.vis:
                self.env.render()
        if self.atari:
            observation = preprocess(observation)
        return observation, tot_reward, done, info

    def reset(self):
        observation = self.env.reset()
        if self.atari:
            return preprocess(observation)
        return observation

    def render(self):
        self.env.render()
