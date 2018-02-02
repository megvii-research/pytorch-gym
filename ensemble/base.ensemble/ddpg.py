import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from model import (Actor, Critic)
from rpm import rpm
from random_process import *

from util import *

criterion = nn.MSELoss()


class DDPG(object):
    def __init__(self, nb_status, nb_actions, args):
        self.num_actor = 3

        self.nb_status = nb_status * args.window_length
        self.nb_actions = nb_actions

        # Create Actor and Critic Network
        net_cfg = {
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
        }

        self.actors = [Actor(self.nb_status, self.nb_actions) for _ in range(self.num_actor)]
        self.actor_targets = [Actor(self.nb_status, self.nb_actions) for _ in
                              range(self.num_actor)]
        self.actor_optims = [Adam(self.actors[i].parameters(), lr=args.prate) for i in range(self.num_actor)]

        self.critic = Critic(self.nb_status, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_status, self.nb_actions, **net_cfg)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.rate)

        for i in range(self.num_actor):
            hard_update(self.actor_targets[i], self.actors[i])  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

        # Create replay buffer
        self.memory = rpm(args.rmsize)  # SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = Myrandom(size=nb_actions)

        # Hyper-parameters
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        # 
        self.epsilon = 1.0
        self.s_t = None  # Most recent state
        self.a_t = None  # Most recent action
        # 
        self.cuda()

    def update_policy(self, train_actor=True):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_batch(self.batch_size)

        # Prepare for the target q batch
        index = np.random.randint(low=0, high=self.num_actor)
        next_q_values = self.critic_target([
            to_tensor(next_state_batch, volatile=True),
            self.actor_targets[index](to_tensor(next_state_batch, volatile=True)),
        ])
        next_q_values.volatile = False

        target_q_batch = to_tensor(reward_batch) + \
                         self.discount * to_tensor((1 - terminal_batch.astype(np.float))) * next_q_values

        # Critic update
        self.critic.zero_grad()
        q_batch = self.critic([to_tensor(state_batch), to_tensor(action_batch)])

        # print(reward_batch, next_q_values*self.discount, target_q_batch, terminal_batch.astype(np.float))
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        sum_policy_loss = 0
        for i in range(self.num_actor):
            self.actors[i].zero_grad()

            policy_loss = -self.critic([
                to_tensor(state_batch),
                self.actors[i](to_tensor(state_batch))
            ])

            policy_loss = policy_loss.mean()
            policy_loss.backward()
            if train_actor:
                self.actor_optims[i].step()
            sum_policy_loss += policy_loss

            # Target update
            soft_update(self.actor_targets[i], self.actors[i], self.tau)

        soft_update(self.critic_target, self.critic, self.tau)

        return -sum_policy_loss / self.num_actor, value_loss

    def cuda(self):
        for i in range(self.num_actor):
            self.actors[i].cuda()
            self.actor_targets[i].cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t1, done):
        self.memory.append([self.s_t, self.a_t, r_t, s_t1, done])
        self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1., 1., self.nb_actions)
        self.a_t = action

        return action

    def select_action(self, s_t, decay_epsilon=True, noise_level=0):
        actions = []
        status = []
        tot_score = []
        for i in range(self.num_actor):
            action = to_numpy(self.actors[i](to_tensor(np.array([s_t]), volatile=True))).squeeze(0)
            noise_level = noise_level * max(self.epsilon, 0)
            action = action + self.random_process.sample() * noise_level
            status.append(s_t)
            actions.append(action)
            tot_score.append(0.)

        scores = self.critic([to_tensor(np.array(status), volatile=True), to_tensor(np.array(actions), volatile=True)])
        for j in range(self.num_actor):
            tot_score[j] += scores.data[j][0]
        best = np.array(tot_score).argmax()

        if decay_epsilon:
            self.epsilon -= self.depsilon

        self.a_t = actions[best]
        return actions[best]

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_status()
