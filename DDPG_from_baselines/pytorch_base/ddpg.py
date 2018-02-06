import torch.nn as nn
from torch.optim import Adam
from running_mean_std import RunningMeanStd
from model import (Actor, Critic)
from random_process import *

from util import *

criterion = nn.MSELoss()


def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std


class DDPG(object):
    def __init__(self, memory, nb_status, nb_actions, action_noise=None,
                 gamma=0.99, tau=0.001, normalize_observations=True,
                 batch_size=128, observation_range=(-5., 5.), action_range=(-1., 1.),
                 actor_lr=1e-4, critic_lr=1e-3):
        self.nb_status = nb_status
        self.nb_actions = nb_actions
        self.action_range = action_range
        self.observation_range = observation_range
        self.normalize_observations = normalize_observations

        self.actor = Actor(self.nb_status, self.nb_actions)
        self.actor_target = Actor(self.nb_status, self.nb_actions)
        self.actor_optim = Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(self.nb_status, self.nb_actions)
        self.critic_target = Critic(self.nb_status, self.nb_actions)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)

        # Create replay buffer
        self.memory = memory  # SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.action_noise = action_noise

        # Hyper-parameters
        self.batch_size = batch_size
        self.tau = tau
        self.discount = gamma

        if self.normalize_observations:
            self.obs_rms = RunningMeanStd()
        else:
            self.obs_rms = None

    def pi(self, obs, apply_noise=True, compute_Q=True):
        obs = np.array([obs])
        action = to_numpy(self.actor(to_tensor(obs))).squeeze(0)
        if compute_Q:
            q = self.critic([to_tensor(obs), to_tensor(action)]).cpu().data
        else:
            q = None

        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            assert noise.shape == action.shape
            action += noise

        action = np.clip(action, self.action_range[0], self.action_range[1])
        return action, q[0][0]

    def store_transition(self, obs0, action, reward, obs1, terminal1):
        self.memory.append(obs0, action, reward, obs1, terminal1)
        if self.normalize_observations:
            self.obs_rms.update(np.array([obs0]))

    def train(self):
        # Get a batch.
        batch = self.memory.sample(batch_size=self.batch_size)

        next_q_values = self.critic_target([
            to_tensor(batch['obs1'], volatile=True),
            self.actor_target(to_tensor(batch['obs1'], volatile=True))])
        next_q_values.volatile = False

        target_q_batch = to_tensor(batch['rewards']) + \
                         self.discount * to_tensor(1 - batch['terminals1'].astype('float32')) * next_q_values

        self.critic.zero_grad()
        q_batch = self.critic([to_tensor(batch['obs0']), to_tensor(batch['actions'])])
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        self.actor.zero_grad()
        policy_loss = -self.critic([to_tensor(batch['obs0']), self.actor(to_tensor(batch['obs0']))]).mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.cpu().data[0], policy_loss.cpu().data[0]

    def initialize(self):
        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

    def update_target_net(self):
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def reset(self):
        if self.action_noise is not None:
            self.action_noise.reset()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()
