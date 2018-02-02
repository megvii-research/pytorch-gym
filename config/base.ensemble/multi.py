class fastenv():
    def __init__(self, env, action_repeat):
        self.action_repeat = action_repeat
        self.q = []
        self.env = env

    def step(self, action):
        tot_reward = 0.
        for i in range(self.action_repeat):
            observation, reward, done, info = self.env.step(action)
            tot_reward += reward
            if done:
                break
        return observation, tot_reward, done, info

    def reset(self):
        tmp = self.env.reset()
        return tmp
