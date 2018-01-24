
import numpy as np
from observation_processor import queue
from util import *

class Evaluator(object):

    def __init__(self, num_episodes, max_episode_length=None):
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.results = np.array([]).reshape(num_episodes,0)

    def __call__(self, env, policy, debug=False, visualize=False, window_length=1):

        episode_memory = queue()
        observation = None
        result = []

        for episode in range(self.num_episodes):

            # reset at the start of episode
            episode_memory.clear()
            observation = env.reset()
            episode_memory.append(observation)
            observation = episode_memory.getObservation(window_length, observation)
            episode_steps = 0
            episode_reward = 0.                
            assert observation is not None

            # start episode
            done = False
            while not done:
                action = policy(observation)
                observation, reward, done, info = env.step(action)
                episode_memory.append(observation)
                observation = episode_memory.getObservation(window_length, observation)
                if self.max_episode_length and episode_steps >= self.max_episode_length - 1:
                    done = True            
                if visualize:
                    env.render()
                # update
                episode_reward += reward
                episode_steps += 1
            if debug: prRed('[Evaluate] reward:{}'.format(episode_reward))
                
            result.append(episode_reward)

        return np.mean(result)
