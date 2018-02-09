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
    def __init__(self, nb_status, nb_actions, args, writer):
        self.clip_actor_grad = args.clip_actor_grad
        self.nb_status = nb_status * args.window_length
        self.nb_actions = nb_actions
        self.writer = writer
        self.select_time = 0
        
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2, 
            'init_method':args.init_method
        }

        self.actor = Actor(self.nb_status, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_status, self.nb_actions, **net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic(self.nb_status, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_status, self.nb_actions, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        #Create replay buffer
        self.memory = rpm(args.rmsize)
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
        self.use_cuda = args.cuda
        # 
        if self.use_cuda: self.cuda()

    def normalize(self, pic):
        pic = pic.swapaxes(0, 2).swapaxes(1, 2)
        return pic

    def update_policy(self, train_actor = True):
        # Sample batch
        state_batch, action_batch, reward_batch, \
            next_state_batch, terminal_batch = self.memory.sample_batch(self.batch_size)

        # Prepare for the target q batch
        if self.pic:
            state_batch = np.array([self.normalize(x) for x in state_batch])
            state_batch = to_tensor(state_batch, volatile=True)
            print('label 1')
            print('size = ', state_batch.shape)
            state_batch = self.cnn(state_batch)
            print('label 2')
            next_state_batch = np.array([self.normalize(x) for x in next_state_batch])
            next_state_batch = to_tensor(next_state_batch, volatile=True)
            next_state_batch = self.cnn(next_state_batch)
            next_q_values = self.critic_target([
                next_state_batch,
                self.actor_target(next_state_batch)
            ])
        else:
            next_q_values = self.critic_target([
                to_tensor(next_state_batch, volatile=True),
                self.actor_target(to_tensor(next_state_batch, volatile=True)),
            ])
        # print('batch of picture is ok')
        next_q_values.volatile = False

        target_q_batch = to_tensor(reward_batch) + \
            self.discount * to_tensor((1 - terminal_batch.astype(np.float))) * next_q_values

        # Critic update
        self.critic.zero_grad()
        if self.pic: self.cnn.zero_grad()

        if self.pic:
            state_batch.volatile = False
            q_batch = self.critic([state_batch, to_tensor(action_batch)])
        else:
            q_batch = self.critic([to_tensor(state_batch), to_tensor(action_batch)])

        # print(reward_batch, next_q_values*self.discount, target_q_batch, terminal_batch.astype(np.float))
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()
        if self.pic: self.cnn_optim.step()

        self.actor.zero_grad()

        if self.pic:
            state_batch.volatile = False
            policy_loss = -self.critic([
                state_batch,
                self.actor(state_batch)
            ])
        else:
            policy_loss = -self.critic([
                to_tensor(state_batch),
                self.actor(to_tensor(state_batch))
            ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()

        if self.clip_actor_grad is not None:
            torch.nn.utils.clip_grad_norm(self.actor.parameters(), float(self.clip_actor_grad))

            if self.writer != None:
                mean_policy_grad = np.array(np.mean([np.linalg.norm(p.grad.data.cpu().numpy().ravel()) for p in self.actor.parameters()]))
                #print(mean_policy_grad)
                self.writer.add_scalar('train/mean_policy_grad', mean_policy_grad, self.select_time)

        if train_actor:
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

    def train(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def cuda(self):
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

    def select_action(self, s_t, decay_epsilon=True, return_fix=False, noise_level=0):
        if self.pic:
            # print(s_t.shape)
            s_t = self.normalize(s_t)
            s_t = self.cnn(to_tensor(np.array([s_t])))
        self.eval()
        # print(s_t.shape)
        if self.pic:
            action = to_numpy(
                self.actor(s_t)
            ).squeeze(0)
        else:
            action = to_numpy(
                self.actor(to_tensor(np.array([s_t])))
            ).squeeze(0)
            
        self.train()
        noise_level = noise_level * max(self.epsilon, 0)
        
        action = action * (1 - noise_level) + (self.random_process.sample() * noise_level)
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon

        if self.use_cuda:
            tmp = nn.Parameter(torch.FloatTensor(np.array([action])).cuda())
        else:
            tmp = nn.Parameter(torch.FloatTensor(np.array([action])))
        optim = Adam([tmp], lr=1e-3)
        if self.train_actions == True:
            for i in range(10):
                optim.zero_grad()
                loss = -self.critic([
                    to_tensor(np.array([s_t])), tmp
                ])
                loss.backward()
                optim.step()
                tmp = torch.clamp(tmp, -1., 1.)

        Q1 = self.critic([
            to_tensor(np.array([s_t]), volatile=True), to_tensor(np.array([action]), volatile=True)
        ])
        Q2 = self.critic([
            to_tensor(np.array([s_t]), volatile=True), tmp
        ])
        if self.writer != None:
            self.writer.add_scalar('train/d_Q', (Q2 - Q1).data.cpu().numpy(), self.select_time)
            self.select_time += 1

        for j in range(self.nb_actions):
            action[j] = tmp[0][j]
        action = np.clip(action, -1., 1.)
        self.a_t = action
        if return_fix:
            return action
        if self.discrete:
            return action.argmax()
        else:
            return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_status()

    def load_weights(self, output, num=1):        
        if output is None: return
        self.actor.load_state_dict(
            torch.load('{}/actor{}.pkl'.format(output, num))
        )
        self.actor_target.load_state_dict(
            torch.load('{}/actor{}.pkl'.format(output, num))
        )
        self.critic.load_state_dict(
            torch.load('{}/critic{}.pkl'.format(output, num))
        )
        self.critic_target.load_state_dict(
            torch.load('{}/critic{}.pkl'.format(output, num))
        )

    def save_model(self, output, num):
        if self.use_cuda:
            self.actor.cpu()
            self.critic.cpu()
        torch.save(
            self.actor.state_dict(),
            '{}/actor{}.pkl'.format(output, num)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic{}.pkl'.format(output, num)
        )
        if self.use_cuda:
            self.actor.cuda()
            self.critic.cuda()
