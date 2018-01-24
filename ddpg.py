import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model import (Actor, Critic)
from rpm import rpm
from random_process import *

from util import *

criterion = nn.MSELoss()

class DDPG(object):
    def __init__(self, nb_states, nb_actions, args, discrete, use_cuda=False):
        
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.discrete = discrete
        
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2, 
            'init_w':args.init_w
        }
        self.actor = Actor(self.nb_states * args.window_length, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states * args.window_length, self.nb_actions, **net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic(self.nb_states * args.window_length, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states * args.window_length, self.nb_actions, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        #Create replay buffer
        self.memory = rpm(args.rmsize) # SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = Myrandom(size=nb_actions)

        # Hyper-parameters
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        # 
        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.use_cuda = use_cuda
        # 
        if self.use_cuda: self.cuda()
        
    def update_policy(self, train_actor = True):
        # Sample batch
        state_batch, action_batch, reward_batch, \
            next_state_batch, terminal_batch = self.memory.sample_batch(self.batch_size)

        # state_batch, action_batch, reward_batch, \
        # next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        next_q_values = self.critic_target([
            to_tensor(next_state_batch, volatile=True),
            self.actor_target(to_tensor(next_state_batch, volatile=True)),
        ])
        next_q_values.volatile = False

        target_q_batch = to_tensor(reward_batch) + \
            self.discount * to_tensor((1 - terminal_batch.astype(np.float))) * next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([ to_tensor(state_batch), to_tensor(action_batch) ])
        
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        self.actor.zero_grad()

        policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        if train_actor == True:
            self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return -policy_loss, value_loss

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        print("use cuda")
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t1, done):
        self.memory.append([self.s_t, self.a_t, r_t, s_t1, done])
        self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1.,1.,self.nb_actions)
        self.a_t = action
        if self.discrete:
            return action.argmax()
        else:
            return action

    def select_action(self, s_t, decay_epsilon=True, return_fix=False, noise_level=1):
        action = to_numpy(
            self.actor(to_tensor(np.array([s_t])))
        ).squeeze(0)
        # print(self.random_process.sample(), action)
        noise_level = noise_level * max(self.epsilon, 0)
        action = action * (1 - noise_level) + (self.random_process.sample() * noise_level)
        # print(max(self.epsilon, 0) * self.random_process.sample() * noise_level, noise_level)
        action = np.clip(action, -1., 1.)
        # print(action)

        if decay_epsilon:
            self.epsilon -= self.depsilon

        self.a_t = action
        if return_fix:
            return action
        if self.discrete:
            return action.argmax()
        else:
            return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )
        self.actor_target.load_state_dict(
            torch.load('{}/actor_target.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )
        self.critic_target.load_state_dict(
            torch.load('{}/critic_target.pkl'.format(output))
        )

    def save_model(self, output):
        if self.use_cuda:
            self.actor.cpu()
            self.actor_target.cpu()
            self.critic.cpu()
            self.critic_target.cpu()
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.actor_target.state_dict(),
            '{}/actor_target.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )
        torch.save(
            self.critic_target.state_dict(),
            '{}/critic_target.pkl'.format(output)
        )
        if self.use_cuda:
            self.actor.cuda()
            self.actor_target.cuda()
            self.critic.cuda()
            self.critic_target.cuda()

    def seed(self,s):
        torch.manual_seed(s)
        if self.use_cuda:
            torch.cuda.manual_seed(s)
