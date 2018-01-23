#!/usr/bin/env python3 

import numpy as np
import argparse
from copy import deepcopy
import torch
import gym
import pybullet
import pybullet_envs
from baselines import deepq

from normalized_env import NormalizedEnv
from pybullet_envs.bullet.racecarGymEnv import RacecarGymEnv
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from evaluator import Evaluator
from ddpg import DDPG
from util import *
from tensorboardX import SummaryWriter
from collections import deque
import threading as th

import signal
import time
from llll import PythonInstance

gym.undo_logger_setup()

writer = SummaryWriter()

def train(num_iterations, agent, evaluate, proc_cnt, validate_steps, output, window_length, max_episode_length=None,
          debug=False, visualize=False, traintimes=None, resume=None, check_thread=False):
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

    global log
    global step
    global episode
    log = 0
    step = 0
    episode = 0
    agent.is_training = True
    threadx = [None for _ in range(proc_cnt)]
    # max_reward = -100000.
    mutex = th.Lock()

    def episode_once(x):
        observation = None
        pi = PythonInstance(open('./sub_process.py').read())
        pi.send(args)
        while True:
            if check_thread: print('main process: send observation!', observation)
            pi.send(observation)
            if observation is None:
                observation = pi.recv()
                if check_thread: print('main process: receive observation!', observation)
                mutex.acquire()
                agent.reset(observation)
                mutex.release()

            # agent pick action
            # mutex protect
            mutex.acquire()
            global step
            if step <= args.warmup and resume is None:
                action = agent.random_action()
            else:
                action = agent.select_action(observation)
            mutex.release()

            if check_thread: print('main process: send action!', action)
            pi.send(action)
            observation2 = pi.recv()
            if check_thread: print('main process: receive observation2!', observation2)
            reward = pi.recv()
            if check_thread: print('main process: receive reward!', reward)
            done = pi.recv()
            if check_thread: print('main process: receive done!', done)

            # mutex protect
            mutex.acquire()
            agent.observe(reward, observation2, done)
            step += 1
            mutex.release()

            observation = deepcopy(observation2)

            flag = pi.recv()
            if check_thread: print('main process: receive flag!', flag)
            if flag:
                episode_reward = pi.recv()
                if check_thread: print('main process: receive episode_reward!', episode_reward)
                pi.kill()
                pi.join()

                # mutex protect
                global log
                global episode
                episode += 1
                for i in range(args.traintimes):
                    mutex.acquire()
                    log += 1
                    if step > args.warmup:
                        Q, value_loss, policy_loss = agent.update_policy()
                        writer.add_scalar('data/Q', Q.data.numpy(), log)
                        writer.add_scalar('data/critic_loss', value_loss.data.numpy(), log)
                        writer.add_scalar('data/actor_loss', policy_loss.data.numpy(), log)
                    mutex.release()

                mutex.acquire()
                if debug: prGreen('#{}: episode_reward:{} steps:{}'.format(episode, episode_reward, step))
                writer.add_scalar('data/reward', episode_reward, log)
                mutex.release()

                pool.append(x)
                break

    pool = deque()
    for i in range(proc_cnt):
        pool.append(i)
    while step < num_iterations:
        time.sleep(0.1)
        if pool:
            xproc = pool.popleft()
            if threadx[xproc] is not None:
                threadx[xproc].join()
            threadx[xproc] = th.Thread(target=episode_once, args=(xproc,))
            threadx[xproc].start()

    for thr in threadx:
        if thr is not None:
            thr.join()


        # reset if it is the start of episode
        # for j in range(proc_cnt):
        #    proc[j].send(observation[j])


        # if observation is None:
        #    observation = env.reset()
        #    episode_memory.append(observation)
        #    observation = episode_memory.getObservation(window_length, observation)
        #    agent.reset(observation)

        # agent pick action ...
        # if step <= args.warmup and resume == None:
        #    action = agent.random_action()
        # else:
        #    action = agent.select_action(observation)
        # if visualize and step > args.warmup:
        #    env.render()
            
        # env response with next_observation, reward, terminate_info

        # print("action = ", action)
        # observation2, reward, done, info = env.step(action)
        # episode_memory.append(observation2)
        # observation2 = episode_memory.getObservation(window_length, observation2)
        
        # print("observation shape = ", np.shape(observation2))
        # print("observation = ", observation2)
        # print("reward = ", reward)
        # exit()       
        # agent observe and update policy
        # agent.observe(reward, observation2, done)

        # [optional] evaluate
        # if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
        #    policy = lambda x: agent.select_action(x, decay_epsilon=False)
            # validate_reward = evaluate(env, policy, debug=False, visualize=False)
        #     if debug: prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))
            # if validate_reward > max_reward and step != 0:
            #    max_reward = validate_reward
        #    if debug: prYellow('save')
        #    agent.save_model(output)

        # update 
        # step += 1
        # episode_steps += 1
        # episode_reward += reward
        # observation = deepcopy(observation2)

        # if done or (episode_steps >= max_episode_length - 1 and max_episode_length): # end of episode
        #    for i in range(traintimes):
        #        log += 1
        #        if step > args.warmup:
        #            Q, value_loss, policy_loss = agent.update_policy()
        #            writer.add_scalar('data/Q', Q.data.numpy(), log)
        #            writer.add_scalar('data/critic_loss', value_loss.data.numpy(), log)
        #            writer.add_scalar('data/actor_loss', policy_loss.data.numpy(), log)
        #    if debug: prGreen('#{}: episode_reward:{} steps:{}'.format(episode,episode_reward,step))
        #    writer.add_scalar('data/reward', episode_reward, log)

            # reset
        #    observation = None
        #    episode_steps = 0
        #    episode_reward = 0.
        #    episode += 1

    sigint_handler(0, 0)

def test(num_episodes, agent, evaluate, model_path, visualize=True, debug=False):

    if model_path == None:
        model_path = 'output/{}-run1'.format(args.env)
    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
        if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    # arguments represent
    parser.add_argument('--env', default='KuKa-v0', type=str, help='open-ai gym environment')
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=200, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--prate', default=1e-4, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=500, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.97, type=float, help='')
    parser.add_argument('--bsize', default=128, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=2000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.02, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.1, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.1, type=float, help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu') 
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=0, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--train_iter', default=1000000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--traintimes', default=100, type=int, help='train times for each episode')
    parser.add_argument('--resume', default=None, type=str, help='Resuming model path for testing')
    parser.add_argument('--output', default='output', type=str, help='Resuming model path for testing')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--discrete', dest='discrete', action='store_true')
    parser.add_argument('--process', default=3, type=int, choices=[i+1 for i in range(10)], help='process num')
    parser.add_argument('--episode_max_time', default=100, type=int)
    parser.add_argument('--check_thread', dest='check_thread', action='store_true')
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay') # TODO
    # parser.add_argument('--cuda', dest='cuda', action='store_true') # TODO

    args = parser.parse_args()
    # StrCat args.output with args.env
    if args.resume == None:
        args.output = get_output_folder(args.output, args.env)
    else:
        args.output = args.resume

# pybullet
    if args.discrete:
        env = gym.make(args.env)
        env.unwrapped
    else:
        env = NormalizedEnv(gym.make(args.env))

    nb_states = env.observation_space.shape[0]
    if args.discrete:
        nb_actions = env.action_space.n
    else:
        nb_actions = env.action_space.shape[0]

    agent = DDPG(nb_states, nb_actions, args, args.discrete)
    # evaluate = Evaluator(args.validate_episodes,
    #    args.validate_steps, max_episode_length=args.max_episode_length)
    evaluate = None
    if args.test is False:
        train(args.train_iter, agent, evaluate, args.process,
              args.validate_steps, args.output, args.window_length, max_episode_length=args.max_episode_length,
              debug=args.debug, visualize=args.vis, traintimes=args.traintimes,
              resume=args.resume, check_thread=args.check_thread)
    else:
        test(args.validate_episodes, agent, evaluate, args.resume,
             visualize=True, debug=args.debug)
