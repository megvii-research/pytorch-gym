import numpy as np
from observation_processor import queue
from util import *

class Evaluator(object):

    def __init__(self, args, bullet=False):
        
        self.bullet = bullet
        self.window_length = args.window_length
        self.validate_episodes = args.validate_episodes
        self.max_episode_length = args.max_episode_length
        self.results = np.array([]).reshape(self.validate_episodes, 0)

    def __call__(self, env, policy, debug=False, visualize=False):
        
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
                if self.max_episode_length and episode_steps >= self.max_episode_length - 1:
                    done = True
                if visualize:
                    if self.bullet:
                        import pybullet
                        pybullet.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=-6.6,cameraTargetPosition=[10,0,0])
                    env.render()
                episode_reward += reward
                episode_steps += 1
            result.append(episode_reward)
        if debug: prRed('[Evaluate] reward:{}'.format(result))
        return result
