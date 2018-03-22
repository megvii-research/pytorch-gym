#!/usr/bin/env python3
import cv2
import random
import numpy as np
import argparse
from copy import deepcopy
import torch
import gym
from normalized_env import *
# from pybullet_envs.bullet.racecarGymEnv import RacecarGymEnv
# from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from evaluator import Evaluator
from ddpg import DDPG
from util import *
from tensorboardX import SummaryWriter
from observation_processor import queue
from multi import fastenv

# from llll import Subprocess

gym.undo_logger_setup()

import time

writer = SummaryWriter()

def train(num_iterations, agent, env, evaluate, bullet):
    fenv = fastenv(env, args.action_repeat, args.vis, args.atari)
    window_length = args.window_length
    validate_interval = args.validate_interval
    save_interval = args.save_interval
    max_episode_length = args.max_episode_length // args.action_repeat
    debug = args.debug
    visualize = args.vis
    traintimes = args.traintimes
    output = args.output
    resume = args.resume
    validate_episodes = args.validate_episodes

    if resume is not None:
        print('load weight')
        agent.load_weights(output)
        agent.memory.load(output)

    time_stamp = 0.
    log = 0
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    episode_num = 0
    episode_memory = queue()
    noise_level = random.uniform(0, 1) / 2.
    save_num = 0
    # validate_num = 0
    
    while step <= num_iterations:
        # reset if it is the start of episode
        if observation is None:
            episode_memory.clear()
            observation = fenv.reset()
            episode_memory.append(observation)
            observation = episode_memory.getObservation(window_length, observation, args.pic)
            agent.reset(observation)

        # agent pick action ...
        if step <= args.warmup and resume is None:
            action = agent.random_action()
        else:
            # print("observation shape:", observation.shape)            
            action = agent.select_action(observation, noise_level=noise_level)
            
        # env response with next_observation, reward, terminate_info
        # print("action = ", action)
        observation, reward, done, info = fenv.step(action)
        episode_memory.append(observation)
        observation = episode_memory.getObservation(window_length, observation, args.pic)
        
        # print("observation = ", observation)
        # print("reward = ", reward)
        # agent observe and update policy
        agent.observe(reward, observation, done)
        # update 
        step += 1
        episode_steps += 1
        episode_reward += reward
        if (done or (episode_steps >= max_episode_length and max_episode_length)): # end of episode
            # [optional] save
            # if args.env == "Paint":
                # writer.add_image(str(episode) + '.png', env.canvas)       
            if step > args.warmup:
                # [optional] evaluate
                if episode > 0 and validate_interval > 0 and episode % validate_interval == 0:
                    validate_reward = evaluate(fenv, agent.select_action, debug=debug, visualize=False)
                    if debug: prRed('Step_{:07d}: mean_reward:{} reward_var:{}'.format(step, np.mean(validate_reward), np.var(validate_reward)))
                    writer.add_scalar('validate/reward', np.mean(validate_reward), step)
                    if args.env == "Paint":
                        writer.add_image(str(step) + '.png', env.canvas)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            for i in range(traintimes):
                if step > args.warmup:
                    log += 1
                    # print('updating', i)
                    Q, value_loss = agent.update_policy()
                    writer.add_scalar('train/Q', Q.data.cpu().numpy(), log)
                    writer.add_scalar('train/critic_loss', value_loss.data.cpu().numpy(), log)
            if debug: prBlack('#{}: train_reward:{:.3f} steps:{} noise_scale:{:.2f} interval_time:{:.2f} train_time:{:.2f}' \
                .format(episode,episode_reward,step,noise_level,train_time_interval,time.time()-time_stamp))
            time_stamp = time.time()
            writer.add_scalar('train/train_reward', episode_reward, episode)
            
            # reset
            noise_level = random.uniform(0, 1) / 2.
            episode_num += 1
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            
    if debug: prRed('[Save model] #{}'.format(save_num))
    agent.save_model(output, save_num)

def test(validate_episodes, agent, env, evaluate, model_path, window_length, visualize=True, debug=False, bullet=False):
    if model_path is None:
        model_path = 'output/{}-run1'.format(args.env)
    agent.load_weights(model_path)
    if debug: prRed('load model from {}'.format(model_path))        
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(validate_episodes):
        validate_reward = evaluate(env, policy, window_length=window_length, visualize=visualize, debug=debug, bullet=bullet)
        if debug: prRed('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    # arguments represent
    parser.add_argument('--env', default='CartPole-v0', type=str, help='open-ai gym environment')
    parser.add_argument('--hidden1', default=512, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=256, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--prate', default=1e-4, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--crate', default=1e-4, type=float)
    
    parser.add_argument('--warmup', default=1000, type=int, help='timestep without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.95, type=float, help='')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=1000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
    parser.add_argument('--action_repeat', default=1, type=int, help='repeat times for each action')
    
    parser.add_argument('--validate_episodes', default=1, type=int, help='how many episode to perform during validation')
    parser.add_argument('--max_episode_length', default=0, type=int, help='')
    parser.add_argument('--validate_interval', default=100, type=int, help='how many episodes to perform a validation')
    parser.add_argument('--save_interval', default=100, type=int, help='how many episodes to save model')
    parser.add_argument('--train_iter', default=2000000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=10000000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--traintimes', default=5, type=int, help='train times for each episode')    
    parser.add_argument('--resume', default=None, type=str, help='Resuming model path for testing')
    parser.add_argument('--clip_actor_grad', default=None, help='Clip the gradient of the actor by norm.')
    parser.add_argument('--output', default='output', type=str, help='Resuming model path for testing')
    parser.add_argument('--init_method', default='uniform', choices=['uniform', 'normal'], type=str, help='Initialization method of params.')

    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--test', action='store_true', help='test or not')
    parser.add_argument('--vis', action='store_true', help='visualize each action or not')
    parser.add_argument('--discrete', dest='discrete', action='store_true', help='the actions are discrete or not')
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--pic', dest='pic', action='store_true', help='picture input or not')
    parser.add_argument('--atari', dest='atari', action='store_true', help='atari picture input or not')
    parser.add_argument('--pic_status', default=512, type=int)
    parser.add_argument('--bn', action='store_true', help='use BatchNorm layers')

    parser.add_argument('--seed', default=-1, type=int, help='')
    
    args = parser.parse_args()

    if args.resume is None:
        args.output = get_output_folder(args.output, args.env)
    else:
        args.output = args.resume

    bullet = ("Bullet" in args.env)
    if bullet:
        import pybullet
        import pybullet_envs

    if args.env == "Paint":
        from env import CanvasEnv
        env = CanvasEnv()
        writer.add_image('circle.png', env.target)
    elif args.env == "KukaGym":
        env = KukaGymEnv(renders=False, isDiscrete=True)
    elif args.env == "LTR":
        from osim.env import RunEnv
        env = RunEnv(visualize=False)
    elif args.discrete:        
        env = gym.make(args.env)
        env = env.unwrapped
    else:
        env = NormalizedEnv(gym.make(args.env))

    # input random seed
    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    # input status count & actions count
    print('observation_space', env.observation_space.shape, 'action_space', env.action_space.shape)
    nb_status = env.observation_space.shape[0]
    if args.discrete:
        nb_actions = env.action_space.n
    else:
        nb_actions = env.action_space.shape[0]
    
    if args.vis:
        if bullet:
            import pybullet
#            pybullet.resetDebugVisualizerCamera \
#                (cameraDistance=10, cameraYaw=0, cameraPitch=-6.6, cameraTargetPosition=[10,0,0])
        env.render()
        
    agent = DDPG(nb_status, nb_actions, args, writer)
    evaluate = Evaluator(args, bullet=bullet)
    
    if args.test is False:
        train(args.train_iter, agent, env, evaluate, bullet=bullet)
    else:
        test(args.validate_episodes, agent, env, evaluate, bullet=bullet)
