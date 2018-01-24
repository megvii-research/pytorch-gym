#!/usr/bin/env python3 
import random
import numpy as np
import argparse
from copy import deepcopy
import torch
import gym
import signal
import pybullet
import pybullet_envs
from normalized_env import NormalizedEnv
from pybullet_envs.bullet.racecarGymEnv import RacecarGymEnv
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from evaluator import Evaluator
from ddpg import DDPG
from util import *
from tensorboardX import SummaryWriter
from observation_processor import queue
from multi import fastenv

from llll import Subprocess

gym.undo_logger_setup()

import time

writer = SummaryWriter()

def train(num_iterations, agent, env, evaluate, validate_interval, output, window_length, max_episode_length=None,
          debug=False, visualize=False, traintimes=None, resume=None):
    if resume is not None:
        print('load weight')
        agent.load_weights(output)
        agent.memory.load(output)

    def sigint_handler(signum, frame):
        print('memory saving...'),
        agent.memory.save(output)
        print('done')
        exit()
    signal.signal(signal.SIGINT, sigint_handler)

    time_stamp = 0.
    log = 0
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    max_reward = -100000.
    episode_num = 0
    episode_memory = queue()
    noise_level = random.uniform(0, 1) / 2.
    while step < num_iterations:
        # reset if it is the start of episode
        if observation is None:
            episode_memory.clear()
            observation = env.reset()
            episode_memory.append(observation)
            observation = episode_memory.getObservation(window_length, observation)
            agent.reset(observation)

        # agent pick action ...
        if step <= args.warmup and resume is None:
            action = agent.random_action()
        else:
            action = agent.select_action(observation, noise_level=noise_level)
            
        # env response with next_observation, reward, terminate_info

        # print("action = ", action)
        observation, reward, done, info = env.step(action)
        episode_memory.append(observation)
        observation = episode_memory.getObservation(window_length, observation)
        
        # print("observation shape = ", np.shape(observation))
        # print("observation = ", observation)
        # print("reward = ", reward)
        # exit()       
        # agent observe and update policy
        agent.observe(reward, observation, done)
        # update 
        step += 1
        episode_steps += 1
        episode_reward += reward

        if done or (episode_steps >= max_episode_length - 1 and max_episode_length): # end of episode

            # [optional] evaluate
            if evaluate is not None and validate_interval > 0 and episode % validate_interval == 0:
                policy = lambda x: agent.select_action(x, decay_epsilon=False, noise_level=0)
                validate_reward = evaluate(env, policy, debug=False, visualize=False, window_length=window_length)
                writer.add_scalar('data/Q', validate_reward, episode / validate_interval)
                if debug: prRed('[Evaluate] Step_{:07d}: mean_reward:{} and save model'.format(step, validate_reward))
                if validate_reward > max_reward and step != 0:
                    max_reward = validate_reward
                agent.save_model(output)
            
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            for i in range(traintimes):
                log += 1
                if step > args.warmup:
                    Q, value_loss = agent.update_policy()
                    writer.add_scalar('data/Q', Q.data.cpu().numpy(), log)
                    writer.add_scalar('data/critic_loss', value_loss.data.cpu().numpy(), log)
            if debug: prBlack('#{}: episode_reward:{:.2f} steps:{} noise:{:.2f} time:{:.2f},{:.2f}' \
                              .format(episode,episode_reward,step,noise_level,train_time_interval,time.time()-time_stamp))
            time_stamp = time.time()
            writer.add_scalar('data/reward', episode_reward, episode)
            
            # reset
            noise_level = random.uniform(0, 1) / 2.
            episode_num += 1
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1

def test(num_episodes, agent, env, evaluate, model_path, window_length, visualize=True, debug=False):

    if model_path is None:
        model_path = 'output/{}-run1'.format(args.env)
    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, window_length, debug=debug, visualize=visualize, save=False)
        if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    # arguments represent
    parser.add_argument('--env', default='CartPole-v0', type=str, help='open-ai gym environment')
    parser.add_argument('--hidden1', default=600, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=3e-4, type=float, help='learning rate')
    parser.add_argument('--prate', default=3e-4, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=1000, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.97, type=float, help='')
    parser.add_argument('--bsize', default=128, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=2000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=2, type=int, help='')
    parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.1, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.1, type=float, help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu') 
    parser.add_argument('--validate_episodes', default=1, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=0, type=int, help='')
    parser.add_argument('--validate_interval', default=10, type=int, help='how many episodes to perform a validate experiment')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--train_iter', default=10000000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=10000000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--traintimes', default=100, type=int, help='train times for each episode')
    parser.add_argument('--action_repeat', default=4, type=int, help='repeat times for each action')
    parser.add_argument('--resume', default=None, type=str, help='Resuming model path for testing')
    parser.add_argument('--output', default='output', type=str, help='Resuming model path for testing')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--discrete', dest='discrete', action='store_true')
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay') # TODO

    args = parser.parse_args()
    # StrCat args.output with args.env
    if args.resume is None:
        args.output = get_output_folder(args.output, args.env)
    else:
        args.output = args.resume

    if args.env == "KukaGym":
        env = KukaGymEnv(renders=False, isDiscrete=True)
    elif args.discrete:
        env = gym.make(args.env)
        env = env.unwrapped
    else:
        env = NormalizedEnv(gym.make(args.env))

    # input random seed
    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    # input states count & actions count
    print(env.observation_space.shape, env.action_space.shape)
    nb_states = env.observation_space.shape[0]
    if args.discrete:
        nb_actions = env.action_space.n
    else:
        nb_actions = env.action_space.shape[0]

    env = fastenv(env, args.action_repeat, args.vis)
    agent = DDPG(nb_states, nb_actions, args, args.discrete, args.cuda)
    evaluate = Evaluator(args.validate_episodes, max_episode_length=args.max_episode_length)

    if args.vis and args.env == 'HalfCheetahBulletEnv-v0':
        env.render()
    
    if args.test is False:
        train(args.train_iter, agent, env, evaluate, 
              args.validate_interval, args.output, args.window_length, max_episode_length=args.max_episode_length,
              debug=args.debug, visualize=args.vis, traintimes=args.traintimes, resume=args.resume)

    else:
        test(args.validate_episodes, agent, env, evaluate, args.resume, args.window_length, 
             visualize=True, debug=args.debug)
