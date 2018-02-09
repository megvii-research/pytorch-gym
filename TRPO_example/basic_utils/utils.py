from collections import defaultdict, OrderedDict
from torch import nn
import numpy as np
import torch
from tabulate import tabulate
import scipy.signal
from tensorboardX import SummaryWriter
import time


def discount(x, gamma):
    """
    Calculate the discounted reward.

    Args:
        x: a list of rewards at each step
        gamma: the discount factor

    Return:
        a list containing the discounted reward
    """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def merge_before_after(info_before, info_after):
    """
    Merge two dicts into one. This is used when updating network weights.
    We merge the update info before update and after update into one dict.

    Args:
        info_before: the dictionary containing the information before updating
        info_after: the dictionary containing the information after updating

    Return:
        info: the merged dictionary
    """
    info = OrderedDict()
    for k in info_before:
        info[k + '_before'] = info_before[k]
        info[k + '_after'] = info_after[k]
    return info


def compute_target(qf, path, gamma, double=False):
    """
    Compute the one step bootstrap target for DQN updating.

    Args:
        qf: the q value function
        path: the data paths for calculation. The result is also saved into it.
        gamma: discount factor
        double: whether to use the double network
    """
    next_observations = path['next_observation']
    not_dones = path['not_done']
    rewards = path['reward'] * (1 - gamma) if gamma < 0.999 else path['reward']
    if not double:
        y_targ = qf.predict(next_observations, target=True).max(axis=1)
    else:
        ty = qf.predict(next_observations).argmax(axis=1)
        y_targ = qf.predict(next_observations, target=True)[np.arange(next_observations.shape[0]), ty]
    path['y_targ'] = y_targ * not_dones * gamma + rewards


class Callback:
    """
    The information printing class.
    """
    def __init__(self, log_file_dir=None):
        self.counter = 0
        self.epi_counter = 0
        self.step_counter = 0
        self.u_stats = dict()
        self.path_info = defaultdict(list)
        self.extra_info = dict()
        self.scores = []
        self.tstart = time.time()
        self.writer = SummaryWriter(log_file_dir)

    def print_table(self):
        """
        Print the saved information, then delete the previous saved information.
        """
        self.counter += 1
        stats = OrderedDict()
        add_episode_stats(stats, self.path_info)
        for d in self.extra_info:
            stats[d] = self.extra_info[d]
        for di in self.u_stats:
            for k in self.u_stats[di]:
                self.u_stats[di][k] = np.mean(self.u_stats[di][k])
        for u in self.u_stats:
            add_prefixed_stats(stats, u, self.u_stats[u])
        stats["TimeElapsed"] = time.time() - self.tstart

        print("************ Iteration %i ************" % self.counter)
        print(tabulate(filter(lambda k: np.asarray(k[1]).size == 1, stats.items())))

        self.scores += self.path_info['episoderewards']
        self.u_stats = dict()
        self.path_info = defaultdict(list)
        self.extra_info = dict()

        return self.counter

    def num_batches(self):
        """
        Gives the number of saved path.

        Return:
            the number of the saved path
        """
        return len(self.path_info['episoderewards'])

    def add_update_info(self, u):
        """
        Save the information from updating.

        Args:
            u: the updating information
        """
        if u is not None:
            for d in u:
                if d[0] not in self.u_stats:
                    self.u_stats[d[0]] = defaultdict(list)
                for k in d[1]:
                    if k[-7:] == '_before':
                        prefix = '_before'
                    elif k[-6:] == '_after':
                        prefix = '_after'
                    else:
                        prefix = ''
                    self.writer.add_scalar('update_data' + prefix + '/' + d[0] + '_' + k, d[1][k], self.step_counter)
                    self.u_stats[d[0]][k].append(d[1][k])
            self.step_counter += 1

    def add_path_info(self, path_info, extra_info, flag='train'):
        """
        Save the game play information.

        Args:
            path_info: the information about rewards
            extra_info: optional additional information
        """
        epi_rewards = []
        path_lens = []
        for p in path_info:
            reward = np.sum(p)
            epi_rewards.append(reward)
            path_lens.append(len(p))
            self.writer.add_scalar('episode_data/reward_' + flag, reward, self.epi_counter)
            self.writer.add_scalar('episode_data/path_length_' + flag, len(p), self.epi_counter)
            self.epi_counter += 1
        for d in extra_info:
            self.writer.add_scalar('extra_info/' + d + '_' + flag, extra_info[d], self.epi_counter)

        if flag == 'train':
            self.path_info['episoderewards'] += epi_rewards
            self.path_info['pathlengths'] += path_lens
            for d in extra_info:
                self.extra_info[d] = extra_info[d]


def add_episode_stats(stats, path_info):
    """
    Calculate the episode statistics.

    Args:
        stats: the dict to which the result is saved
        path_info: the path information for calculation
    """
    episoderewards = np.array(path_info['episoderewards'])
    pathlengths = np.array(path_info['pathlengths'])
    len_paths = len(episoderewards)

    stats["NumEpBatch"] = len_paths
    stats["EpRewMean"] = episoderewards.mean()
    stats["EpRewSEM"] = episoderewards.std() / np.sqrt(len_paths)
    stats["EpRewMax"] = episoderewards.max()
    stats["EpRewMin"] = episoderewards.min()
    stats["EpLenMean"] = pathlengths.mean()
    stats["EpLenMax"] = pathlengths.max()
    stats["EpLenMin"] = pathlengths.min()
    stats["RewPerStep"] = episoderewards.sum() / pathlengths.sum()

    return stats


def add_prefixed_stats(stats, prefix, d):
    """
    Add prefixes to the keys of one dictionary and save the processed terms to another dictionary.

    Args:
        stats: the dict to which the result is saved
        prefix: the prefix to be added
        d: the source dictionary
    """
    if d is not None:
        for k in d:
            stats[prefix + "_" + k] = d[k]


use_cuda = torch.cuda.is_available()


def Variable(tensor, *args, **kwargs):
    """
    The augmented Variable() function which automatically applies cuda() when gpu is available.
    """
    if use_cuda:
        return torch.autograd.Variable(tensor, *args, **kwargs).cuda()
    else:
        return torch.autograd.Variable(tensor, *args, **kwargs)


def np_to_var(nparray):
    """
    Change a numpy variable to a Variable.
    """
    assert isinstance(nparray, np.ndarray)
    return torch.autograd.Variable(torch.from_numpy(nparray).float())


def pre_process_path(paths, keys):
    """
    Pre process the paths into torch Variables.

    Args:
        paths: paths to be processed
        keys: select the keys in the path to be processed

    Return:
        new_path: the processed
    """
    new_path = defaultdict(list)
    for k in keys:
        new_path[k] = np.concatenate([path[k] for path in paths])
        new_path[k] = np_to_var(np.array(new_path[k]))
        if len(new_path[k].size()) == 1:
            new_path[k] = new_path[k].view(-1, 1)
    return new_path


def get_flat_params_from(model):
    """
    Get the flattened parameters of the model.

    Args:
        model: the model from which the parameters are derived

    Return:
        flat_param: the flattened parameters
    """
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    """
    Set the flattened parameters back to the model.

    Args:
        model: the model to which the parameters are set
        flat_params: the flattened parameters to be set
    """
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grads_from(model):
    """
    Get the flattened gradients from the model.

    Args:
        model: the model from which the parameters are derived

    Return:
        flat_param: the flattened parameters
    """
    grads = []
    for param in model.parameters():
        grads.append(param.grad.data.view(-1))
    flat_grads = torch.cat(grads)
    return flat_grads


def set_flat_grads_to(model, flat_grads):
    """
    Set the flattened gradients to the model.

    Args:
        model: the model to which the gradients are set
        flat_grads: the flattened gradients to be set
    """
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.grad = Variable(flat_grads[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def turn_into_cuda(var):
    """
    Change a variable or tensor into cuda.

    Args:
        var: the variable to be changed

    Return:
        the changed variable
    """
    return var.cuda() if use_cuda else var


def log_gamma(xx):
    """
    The log gamma function.
    """
    gamma_coeff = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ]
    magic1 = 1.000000000190015
    magic2 = 2.5066282746310005
    x = xx - 1.0
    t = x + 5.5
    t = t - (x + 0.5) * torch.log(t)
    ser = Variable(torch.ones(x.size())) * magic1
    for c in gamma_coeff:
        x = x + 1.0
        ser = ser + torch.pow(x / c, -1)
    return torch.log(ser * magic2) - t


def digamma(xx):
    """
    The digamma function.
    """
    gamma_coeff = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ]
    magic1 = 1.000000000190015
    x = xx - 1.0
    t = 5/(x+5.5) - torch.log(x+5.5)
    ser = Variable(torch.ones(x.size()), requires_grad=True) * magic1
    ser_p = Variable(torch.zeros(x.size()), requires_grad=True)
    for c in gamma_coeff:
        x = x + 1.0
        ser = ser + torch.pow(x / c, -1)
        ser_p = ser_p - c/(x*x)
    return ser_p/ser - t


def merge_dict(path, single_trans):
    """
    Merge a single transition into the total path.
    """
    for k in single_trans:
        path[k] += single_trans[k]
    return 1 - single_trans['not_done'][0]


def sign(k_id):
    return -1. if k_id % 2 == 0 else 1.

