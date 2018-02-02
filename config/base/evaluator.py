import numpy as np
from observation_processor import queue
from util import *


class Evaluator(object):
    def __init__(self, args):
        self.window_length = args.window_length
        self.validate_episodes = args.validate_episodes
        self.results = np.array([]).reshape(self.validate_episodes, 0)

    def __call__(self, env, policy):
        episode_memory = queue()
        observation = None
        result = []
        for episode in range(self.validate_episodes):

            # reset at the start of episode
            episode_memory.clear()
            observation = env.reset()
            episode_memory.append(observation)
            observation = episode_memory.getObservation(self.window_length, observation)
            episode_steps = 0
            episode_reward = 0.
            assert observation is not None

            # start episode
            done = False
            while not done:
                action = policy(observation)
                observation, reward, done, info = env.step(action)
                episode_memory.append(observation)
                observation = episode_memory.getObservation(self.window_length, observation)
                episode_reward += reward
                episode_steps += 1
            result.append(episode_reward)
        return result
