from llll import sbc
import gym
from normalized_env import NormalizedEnv
import numpy as np
from observation_processor import *

sbc = sbc()
args = sbc.recv()
if args.discrete:
    env = gym.make(args.env)
    env.unwrapped
else:
    env = NormalizedEnv(gym.make(args.env))

if args.seed > 0:
    np.random.seed(args.seed)
    env.seed(args.seed)

sbc.send(env.observation_space.shape[0])
if args.discrete:
    sbc.send(env.action_space.n)
else:
    sbc.send(env.action_space.shape[0])

# start episode

episode = episode_steps = 0
episode_reward = 0.
episode_memory = queue()

while True:
    observation = sbc.recv()
    if observation is None:
        observation = env.reset()
        episode_memory.append(observation)
        observation = episode_memory.getObservation(args.window_length, observation)
        sbc.send(observation)

    action = sbc.recv()
    observation2, reward, done, info = env.step(action)
    episode_memory.append(observation2)
    observation2 = episode_memory.getObservation(args.window_length, observation2)

    sbc.send(observation2)
    sbc.send(reward)
    sbc.send(done)
    # sbc.send(info)

    episode_steps += 1
    episode_reward += reward

    flag = False

    if done or (episode_steps >= args.max_episode_length - 1 and args.max_episode_length):  # end of episode
        # for i in range(args.traintimes):
        #     log += 1
        #     if step > args.warmup:
        #         Q, value_loss, policy_loss = agent.update_policy()
        #        writer.add_scalar('data/Q', Q.data.numpy(), log)
        #        writer.add_scalar('data/critic_loss', value_loss.data.numpy(), log)
        #        writer.add_scalar('data/actor_loss', policy_loss.data.numpy(), log)
        # if debug: prGreen('#{}: episode_reward:{} steps:{}'.format(episode, episode_reward, step))
        # writer.add_scalar('data/reward', episode_reward, log)

        # reset
        observation = None
        episode_steps = 0
        episode_reward = 0.
        episode += 1
        flag = True

    sbc.send(flag)

