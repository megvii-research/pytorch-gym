#!/usr/bin/env python3 
import argparse
import gym
import pybullet
import pybullet_envs
from ddpg import DDPG
from util import *
from memory import Memory
from random_process import *
from collections import deque
import time
import logger


def train(env, nb_epochs, nb_epoch_cycles, normalize_observations, actor_lr, critic_lr, action_noise,
          gamma, nb_train_steps, nb_rollout_steps, batch_size, memory, tau=0.01):

    max_action = env.action_space.high
    agent = DDPG(memory, env.observation_space.shape[0], env.action_space.shape[0],
                 gamma=gamma, tau=tau,
                 normalize_observations=normalize_observations,
                 batch_size=batch_size, action_noise=action_noise,
                 actor_lr=actor_lr, critic_lr=critic_lr,
                 )
    if USE_CUDA:
        agent.cuda()
    # Set up logging stuff only for a single worker.
    step = 0
    episode = 0
    episode_rewards_history = deque(maxlen=100)
    # Prepare everything.

    agent.reset()
    obs = env.reset()
    done = False
    episode_reward = 0.
    episode_step = 0
    episodes = 0
    t = 0

    epoch = 0
    start_time = time.time()

    epoch_episode_rewards = []
    epoch_episode_steps = []
    epoch_start_time = time.time()
    epoch_actions = []
    epoch_qs = []
    epoch_episodes = 0
    for epoch in range(nb_epochs):
        for cycle in range(nb_epoch_cycles):
            # Perform rollouts.
            for t_rollout in range(nb_rollout_steps):
                # Predict next action.
                action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                assert action.shape == env.action_space.shape

                # Execute next action.
                assert max_action.shape == action.shape
                new_obs, r, done, info = env.step(max_action * action)
                t += 1
                episode_reward += r
                episode_step += 1

                # Book-keeping.
                epoch_actions.append(action)
                epoch_qs.append(q)
                agent.store_transition(obs, action, r, new_obs, done)
                obs = new_obs

                if done:
                    # Episode done.
                    epoch_episode_rewards.append(episode_reward)
                    episode_rewards_history.append(episode_reward)
                    epoch_episode_steps.append(episode_step)
                    episode_reward = 0.
                    episode_step = 0
                    epoch_episodes += 1
                    episodes += 1

                    agent.reset()
                    obs = env.reset()

            # Train.
            epoch_actor_losses = []
            epoch_critic_losses = []
            for t_train in range(nb_train_steps):
                cl, al = agent.train()
                epoch_critic_losses.append(cl)
                epoch_actor_losses.append(al)
                agent.update_target_net()

        # Log stats.
        # XXX shouldn't call np.mean on variable length lists
        duration = time.time() - start_time
        combined_stats = dict()
        combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
        combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
        combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
        combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
        combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
        combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
        combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
        combined_stats['total/duration'] = duration
        combined_stats['total/steps_per_second'] = float(t) / float(duration)
        combined_stats['total/episodes'] = episodes
        combined_stats['rollout/episodes'] = epoch_episodes
        combined_stats['rollout/actions_std'] = np.std(epoch_actions)

        # Total statistics.
        combined_stats['total/epochs'] = epoch + 1
        combined_stats['total/steps'] = t

        for key in sorted(combined_stats.keys()):
            logger.record_tabular(key, combined_stats[key])
        logger.dump_tabular()
        logger.info('')


def run(env_id, seed, noise_type, **kwargs):
    # Create envs.
    env = gym.make(env_id)

    # Parse noise_type
    action_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                                        sigma=float(stddev) * np.ones(nb_actions))

    # Configure components.
    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)

    # Seed everything to make things reproducible.
    logger.info('seed={}, logdir={}'.format(seed, logger.get_dir()))
    if seed > 0:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        env.seed(seed)
        torch.cuda.manual_seed(seed)

    start_time = time.time()
    train(env=env, action_noise=action_noise, memory=memory, **kwargs)
    env.close()
    logger.info('total runtime: {}s'.format(time.time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env-id', type=str, default='HalfCheetah-v1')
    boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=9876)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--nb-epochs', type=int, default=500)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    parser.add_argument('--nb-train-steps', type=int, default=50)
    parser.add_argument('--nb-rollout-steps', type=int, default=100)
    parser.add_argument('--noise-type', type=str, default='ou_0.2')  # choices are ou_xx, normal_xx, none
    args = parser.parse_args()

    dict_args = vars(args)
    return dict_args


if __name__ == '__main__':
    args = parse_args()
    logger.configure(dir='/home/nichengzhuo/ddpg_exps_new/results/base.no_mpi.modify/')
    # Run actual script.
    run(**args)
