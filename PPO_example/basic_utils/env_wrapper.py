import gym
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from collections import deque
from gym.spaces import Box
from pybullet_envs.bullet.racecarGymEnv import RacecarGymEnv


class Scaler(object):
    """
    A class recording the mean and variance of the observation.
    """

    def __init__(self, obs_dim):
        """
        Args:
            obs_dim: dim of the observation
        """
        self.vars = np.zeros(obs_dim)
        self.means = np.zeros(obs_dim)
        self.m = 0
        self.first_pass = True

    def update(self, x):
        """
        Update the stored statistics with the new paths.

        Args:
            x: new generated paths
        """
        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
                         np.square(new_means))
            self.vars = np.maximum(0.0, self.vars)
            self.means = new_means
            self.m += n

    def get(self):
        """
        Return the stored statistics.

        Return:
            the scale and mean for the data.
        """
        return 1 / (np.sqrt(self.vars) + 0.1) / 3, self.means


class Vec_env_wrapper:
    """
    A wrapper for the environments whose observation is vector.
    """

    def __init__(self, name, consec_frames, running_stat, seed=None):
        """
        Args:
            name: name of the game
            consec_frames: number of observations to concatenate together
            running_stat: whether to use normalization or not
        """
        self.env = gym.make(name)
        if seed is not None:
            self.env.seed(seed)
        self.name = name
        self.consec_frames = consec_frames
        self.states = deque(maxlen=self.consec_frames)
        self.observation_space = Box(shape=(self.env.observation_space.shape[0] * self.consec_frames + 1,), low=0,
                                     high=1)
        self.observation_space_sca = Box(shape=(self.env.observation_space.shape[0] * self.consec_frames,), low=0,
                                         high=1)
        self.action_space = self.env.action_space

        self.timestep = 0
        self.running_stat = running_stat
        self.offset = None
        self.scale = None

    def set_scaler(self, scales):
        """
        Set the running stats.

        Args:
            scales: the stats to be set
        """
        self.offset = scales[1]
        self.scale = scales[0]

    def _normalize_ob(self, ob):
        """
        Make observation normalization.

        Args:
            ob: observation to be normalized
        """
        if not self.running_stat:
            return ob
        return (ob - self.offset) * self.scale

    def reset(self):
        """
        Reset the environment.

        Return:
            history_normalized: the concatenated and normalized observations
            info: a dict containing the raw observations
        """
        ob = self.env.reset()
        self.timestep = 0
        for i in range(self.consec_frames):
            self.states.append(ob)
        history = np.concatenate(self.states, axis=-1)
        history_normalized = self._normalize_ob(history)
        history_normalized = np.append(history_normalized, [self.timestep])

        info = {"observation_raw": history}
        return history_normalized, info

    def step(self, action):
        """
        Take one step in the environment.

        Args:
            action: the action to take

        Return:
            history_normalized: the concatenated and normalized observations
            r: reward
            done: whether the game is over
            info: a dict containing the raw observations and raw rewards.
        """
        ob, r, done, info = self.env.step(action)
        self.timestep += 1e-3
        self.states.append(ob)
        history = np.concatenate(self.states, axis=-1)
        history_normalized = self._normalize_ob(history)
        history_normalized = np.append(history_normalized, [self.timestep])
        info["reward_raw"] = r
        info["observation_raw"] = history
        return history_normalized, r, done, info

    def render(self):
        """
        Display the game.
        """
        self.env.render()

    def close(self):
        """
        Close the environment.
        """
        self.env.close()
