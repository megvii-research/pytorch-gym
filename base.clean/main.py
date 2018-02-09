#!/usr/bin/env python3 
import random
import numpy as np
import argparse
import torch
import gym
from normalized_env import NormalizedEnv
from multi import fastenv
from ddpg import DDPG
from util import *
from tensorboardX import SummaryWriter
from observation_processor import queue
import sys


gym.undo_logger_setup()

import time


def train(num_iterations, agent, env):
    fenv = fastenv(env, args.action_repeat)
    window_length = args.window_length
    save_interval = args.save_interval
    max_episode_length = args.max_episode_length
    debug = args.debug
    output = args.output

    time_stamp = 0.
    log = 0
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    episode_num = 0
    episode_memory = queue()
    noise_level = args.noise_level * random.uniform(0, 1) / 2.
    save_num = 0
    # validate_num = 0
    
    while step <= num_iterations:
        # reset if it is the start of episode
        if observation is None:
            episode_memory.clear()
            observation = fenv.reset()
            episode_memory.append(observation)
            observation = episode_memory.getObservation(window_length, observation)
            agent.reset(observation)

        # agent pick action ...
        if step <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation, noise_level=noise_level)
            # print('step = ', step)
            
        # env response with next_observation, reward, terminate_info
        observation, reward, done, info = fenv.step(action)
        episode_memory.append(observation)
        observation = episode_memory.getObservation(window_length, observation)
        
        # agent observe and update policy
        agent.observe(reward, observation, done)
        # update 
        step += 1
        episode_steps += 1
        episode_reward += reward
        if (done or (episode_steps >= max_episode_length and max_episode_length)): # end of episode
            # [optional] save
            if step > args.warmup:
                if episode > 0 and save_interval > 0 and episode % save_interval == 0:
                    save_num += 1
                    if debug: prRed('[Save model] #{} in {}'.format(save_num, args.output))
                    agent.save_model(output, save_num)

            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            for i in range(episode_steps):
                if step > args.warmup:
                    log += 1
                    # print('updating', i)
                    Q, value_loss = agent.update_policy()
                    writer.add_scalar('train/Q', Q.data.cpu().numpy(), log)
                    writer.add_scalar('train/critic_loss', value_loss.data.cpu().numpy(), log)

            if debug: prBlack('#{}: train_reward:{:.3f} steps:{} real noise_level:{:.2f} interval_time:{:.2f} train_time:{:.2f}' \
                .format(episode,episode_reward,step,noise_level,train_time_interval,time.time()-time_stamp))
            time_stamp = time.time()
            writer.add_scalar('train/train_reward', episode_reward, episode)
            
            # reset
            noise_level = args.noise_level * random.uniform(0, 1) / 2.
            episode_num += 1
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    # arguments represent
    parser.add_argument('--env', default='CartPole-v0', type=str, help='open-ai gym environment')
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--prate', default=1e-4, type=float, help='policy net learning rate (only for DDPG)')
    
    parser.add_argument('--warmup', default=1000, type=int, help='timestep without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.9, type=float, help='')
    parser.add_argument('--batch_size', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=1000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=3, type=int, help='')
    parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
    parser.add_argument('--action_repeat', default=4, type=int, help='repeat times for each action')
    
    parser.add_argument('--max_episode_length', default=0, type=int, help='')
    parser.add_argument('--save_interval', default=100, type=int, help='how many episodes to save model')
    parser.add_argument('--train_iter', default=2000000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=10000000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--noise_level', default=1, type=float, help='Level of noise to add to actions.')
    parser.add_argument('--clip_actor_grad', default=None, help='Clip the gradient of the actor by norm.')
    parser.add_argument('--output', default='output', type=str, help='Resuming model path for testing')
    parser.add_argument('--init_method', default='uniform', choices=['uniform', 'normal'], type=str, help='Initialization method of params.')

    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--cuda', dest='cuda', action='store_true')

    parser.add_argument('--seed', default=-1, type=int, help='')
    
    args = parser.parse_args()

    args.output = get_output_folder(args.output, args.env)

    if args.debug:
        print('Writing to {}'.format(args.output))

    writer = SummaryWriter(args.output)
    with open(os.path.join(args.output, 'cmdline.txt'), 'a') as f:
        f.write(' '.join(sys.argv) + '\n')

    bullet = ("Bullet" in args.env)
    if bullet:
        import pybullet
        import pybullet_envs
        
    env = NormalizedEnv(gym.make(args.env))

    # input random seed
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        env.seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    # input status count & actions count
    print('observation_space', env.observation_space.shape, 'action_space', env.action_space.shape)
    nb_status = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]
    
    agent = DDPG(nb_status, nb_actions, args, writer)

    train(args.train_iter, agent, env)
