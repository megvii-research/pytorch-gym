import torch.multiprocessing as mp
from torch.multiprocessing import Queue

from basic_utils.env_wrapper import Scaler
from models.agents import *


class Parallel_Path_Data_Generator:
    def __init__(self,
                 agent,
                 env,
                 n_worker,
                 path_num,
                 action_repeat,
                 render):
        """
        Generate several whole paths.

        Args:
            agent: the agent for action selection
            env: the environment
            n_worker: the number of parallel workers
            path_num: number of paths to return at every call
            action_repeat: number of repeated actions
            render: whether display the game or not
        """
        mp.set_start_method('spawn')

        self.n_worker = n_worker
        self.workers = []
        self.senders = []
        self.receivers = []
        self.scaler = Scaler(env.observation_space_sca.shape)
        self.agent = agent
        for i in range(self.n_worker):
            s = Queue()
            r = Queue()
            self.senders.append(s)
            self.receivers.append(r)
            self.workers.append(
                mp.Process(target=path_rollout, args=(agent, env, path_num, r, s, render, action_repeat, i)))
        for worker in self.workers:
            worker.start()

    def __call__(self, num_episode=None):
        """
        Get a fixed number of paths.

        return:
            complete_paths: a list containing several dicts which contains the rewards,
                            actions, observations and not_dones
            path_info: rewards of each step in each path
            extra_info: extra information
        """
        episode_counter = 0
        paths = []
        for i in range(self.n_worker):
            paths.append(defaultdict(list))

        while (num_episode is None) or episode_counter < num_episode:
            counter = 0
            complete_paths = []
            params = self.agent.get_params()
            for index in range(self.n_worker):
                if self.senders[index].empty():
                    self.senders[index].put((params, self.scaler.get()))

            while counter != self.n_worker:
                for i in range(self.n_worker):
                    while not self.receivers[i].empty():
                        single_trans = self.receivers[i].get()
                        if single_trans is None:
                            counter += 1
                        else:
                            done = merge_dict(paths[i], single_trans)
                            if done:
                                path = {k: np.array(paths[i][k]) for k in paths[i]}
                                self.scaler.update(path['observation_raw'])
                                del path['observation_raw']
                                complete_paths.append(path)
                                paths[i] = defaultdict(list)

                if counter == self.n_worker:
                    path_info = [p['reward_raw'] for p in complete_paths]
                    extra_info = {}
                    yield complete_paths, path_info, extra_info


def path_rollout(agent,
                 env,
                 path_num,
                 require_q,
                 recv_q,
                 render,
                 action_repeat,
                 process_id=0):
    """
    Generates several paths for each worker.

    Args:
        agent: the agent for action selection
        env: the environment
        path_num: number of paths to return at every call
        require_q: a queue to put the generated data
        recv_q: a queue to get the params
        render: whether display the game or not
        action_repeat: number of repeated actions
        process_id: id of the worker
    """
    params, scale = recv_q.get()
    agent.set_params(params)

    single_data = defaultdict(list)
    count = 0
    while True:
        env.set_scaler(scale)
        ob, info = env.reset()
        now_repeat = 0
        for k in info:
            single_data[k].append(info[k])
        done = False
        while not done:
            if render and process_id == 0:
                env.render()
            single_data["observation"].append(ob)
            if now_repeat == 0:
                action = agent.act(ob.reshape((1,) + ob.shape))[0]
            now_repeat = (now_repeat + 1) % action_repeat
            single_data["action"].append(action)
            ob, rew, done, info = env.step(agent.process_act(action))
            single_data["next_observation"].append(ob)
            for k in info:
                single_data[k].append(info[k])
            single_data["reward"].append(rew)
            single_data["not_done"].append(1 - done)

            require_q.put(single_data)
            single_data = defaultdict(list)
        count += 1
        if count >= path_num:
            require_q.put(None)
            params, scale = recv_q.get()
            agent.set_params(params)
            count = 0
