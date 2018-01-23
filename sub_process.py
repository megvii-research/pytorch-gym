from llll import sbc
import gym
from normalized_env import NormalizedEnv
import numpy as np
from observation_processor import *
import signal

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

# start episode

episode_steps = 0
episode_reward = 0.
episode_memory = queue()


def sigalarm_handler(signum, frame):
    exit()


signal.signal(signal.SIGALRM, sigalarm_handler)


while True:
    signal.alarm(args.episode_max_time)

    observation = sbc.recv()
    if args.check_thread: print('subprocess: receive observation!', observation)
    if observation is None:
        observation = env.reset()
        episode_memory.append(observation)
        observation = episode_memory.getObservation(args.window_length, observation)
        if args.check_thread: print('subprocess: send observation!', observation)
        sbc.send(observation)

    action = sbc.recv()
    if args.vis:
        env.render()
    if args.check_thread: print('subprocess: receive action!', action)
    observation2, reward, done, info = env.step(action)
    episode_memory.append(observation2)
    observation2 = episode_memory.getObservation(args.window_length, observation2)

    if args.check_thread: print('subprocess: send observation2!', observation2)
    sbc.send(observation2)
    if args.check_thread: print('subprocess: send reward!', reward)
    sbc.send(reward)
    if args.check_thread: print('subprocess: send done!', done)
    sbc.send(done)
    # sbc.send(info)

    # update
    episode_steps += 1
    episode_reward += reward


    if done or (episode_steps >= args.max_episode_length - 1 and args.max_episode_length):  # end of episode
        if args.check_thread: print('flag is True')
        sbc.send(True)
        # for i in range(args.traintimes):
        #     log += 1
        #     if step > args.warmup:
        #         Q, value_loss, policy_loss = agent.update_policy()
        #        writer.add_scalar('data/Q', Q.data.numpy(), log)
        #        writer.add_scalar('data/critic_loss', value_loss.data.numpy(), log)
        #        writer.add_scalar('data/actor_loss', policy_loss.data.numpy(), log)
        # if debug: prGreen('#{}: episode_reward:{} steps:{}'.format(episode, episode_reward, step))
        # writer.add_scalar('data/reward', episode_reward, log)

        if args.check_thread: print('subprocess: send episode_reward!', episode_reward)
        sbc.send(episode_reward)

    else:
        if args.check_thread: print('flag is False')
        sbc.send(False)

