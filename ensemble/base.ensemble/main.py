#!/usr/bin/env python3
import os

used_gpu = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = used_gpu

import random
import numpy as np
import argparse
import gym
from normalized_env import NormalizedEnv
from pybullet_envs.bullet.racecarGymEnv import RacecarGymEnv
from evaluator import Evaluator
from ddpg import DDPG
from util import *
from tensorboardX import SummaryWriter
from observation_processor import queue
from multi import fastenv


gym.undo_logger_setup()
writer = SummaryWriter('/home/nichengzhuo/ddpg_exps_new/results/base.ensemble')


def train(num_iterations, agent, env, evaluate):
    fenv = fastenv(env, args.action_repeat)
    window_length = args.window_length
    validate_interval = args.validate_interval

    log = 0
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    episode_num = 0
    episode_memory = queue()
    noise_level = random.uniform(0, 1) / 2.
    
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

        observation, reward, done, info = fenv.step(action)
        episode_memory.append(observation)
        observation = episode_memory.getObservation(window_length, observation)
        agent.observe(reward, observation, done)

        # update 
        step += 1
        episode_steps += 1
        episode_reward += reward
        if done:
            if step > args.warmup:
                # [optional] evaluate
                if episode > 0 and validate_interval > 0 and episode % validate_interval == 0:
                    validate_reward = evaluate(env, agent.select_action)
                    writer.add_scalar('validate/reward', np.mean(validate_reward), step)
            for i in range(episode_steps):
                if step > args.warmup:
                    log += 1
                    Q, value_loss = agent.update_policy()
                    writer.add_scalar('train/Q', Q.data.cpu().numpy(), log)
                    writer.add_scalar('train/critic_loss', value_loss.data.cpu().numpy(), log)
            writer.add_scalar('train/train_reward', episode_reward, episode)
            
            # reset
            noise_level = random.uniform(0, 1) / 2.
            episode_num += 1
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    # arguments represent
    parser.add_argument('--env', default='HalfCheetahBulletEnv-v0', type=str, help='open-ai gym environment')
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--prate', default=1e-4, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--crate', default=1e-4, type=float)
    parser.add_argument('--warmup', default=1000, type=int, help='timestep without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.9, type=float, help='')
    parser.add_argument('--batch_size', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=1000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=3, type=int, help='')
    parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
    parser.add_argument('--action_repeat', default=4, type=int, help='repeat times for each action')
    parser.add_argument('--validate_episodes', default=1, type=int, help='how many episode to perform during validation')
    parser.add_argument('--validate_interval', default=10, type=int, help='how many episodes to perform a validation')
    parser.add_argument('--train_iter', default=2000000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=10000000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=666666, type=int, help='')
    args = parser.parse_args()

    env = NormalizedEnv(gym.make(args.env))

    # input random seed
    if args.seed > 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        env.seed(args.seed)
        random.seed(args.seed)

    # input status count & actions count
    print('observation_space', env.observation_space.shape, 'action_space', env.action_space.shape)
    nb_status = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]

    agent = DDPG(nb_status, nb_actions, args)
    evaluate = Evaluator(args)

    if args.seed > 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        env.seed(args.seed)
        random.seed(args.seed)

    train(args.train_iter, agent, env, evaluate)
