from torch import optim

from basic_utils.utils import *


# ================================================================
# Abstract Classes
# ================================================================
class Updater:
    """
    This is the abstract class of the policy updater.
    """
    def __call__(self, path):
        """
        Update the network weights.

        Args:
            path: a dict consisting the data for updating

        Return:
            a dict containing the information about the process
        """
        raise NotImplementedError


class Optimizer:
    """
    This is the abstract class of the value optimizer.
    """

    def __call__(self, path):
        """
        Update the network weights.

        Args:
            path: a dict consisting the data for updating

        Return:
            a dict containing the information about the process
        """
        raise NotImplementedError


# ================================================================
# Ppo Updater
# ================================================================
class PPO_adapted_Updater(Updater):
    def __init__(self, net, probtype, beta, kl_cutoff_coeff, kl_target, epochs, lr, beta_range, adj_thres,
                 get_info=True):
        self.net = net
        self.probtype = probtype
        self.beta = beta  # dynamically adjusted D_KL loss multiplier
        self.eta = kl_cutoff_coeff
        self.kl_targ = kl_target
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.get_info = get_info
        self.beta_upper = beta_range[1]
        self.beta_lower = beta_range[0]
        self.beta_adj_thres = adj_thres

    def _derive_info(self, observes, actions, advantages, old_prob):
        prob = self.net(observes)
        logp = self.probtype.loglikelihood(actions, prob)
        logp_old = self.probtype.loglikelihood(actions, old_prob)
        kl = self.probtype.kl(old_prob, prob).mean()
        surr = -(advantages * (logp - logp_old).exp()).mean()
        loss = surr + self.beta * kl

        if kl.data[0] - 2.0 * self.kl_targ > 0:
            loss += self.eta * (kl - 2.0 * self.kl_targ).pow(2)

        entropy = self.probtype.entropy(prob).mean()
        info = {'loss': loss.data[0], 'surr': surr.data[0], 'kl': kl.data[0], 'entropy': entropy.data[0],
                'beta_pen': self.beta, 'lr': self.lr}

        return info

    def __call__(self, path):
        observes = turn_into_cuda(path["observation"])
        actions = turn_into_cuda(path["action"])
        advantages = turn_into_cuda(path["advantage"])

        old_prob = self.net(observes).detach()

        if self.get_info:
            info_before = self._derive_info(observes, actions, advantages, old_prob)

        for e in range(self.epochs):
            prob = self.net(observes)
            logp = self.probtype.loglikelihood(actions, prob)
            logp_old = self.probtype.loglikelihood(actions, old_prob)
            kl = self.probtype.kl(old_prob, prob).mean()
            surr = -(advantages * (logp - logp_old).exp()).mean()
            loss = surr + self.beta * kl

            if kl.data[0] - 2.0 * self.kl_targ > 0:
                loss += self.eta * (kl - 2.0 * self.kl_targ).pow(2)

            self.net.zero_grad()
            loss.backward()
            self.optimizer.step()

            prob = self.net(observes)
            kl = self.probtype.kl(old_prob, prob).mean()
            if kl.data[0] > self.kl_targ * 4:
                break
        if kl.data[0] > self.kl_targ * self.beta_adj_thres[1]:
            if self.beta_upper > self.beta:
                self.beta = self.beta * 1.5
            if self.beta > self.beta_upper / 1.5:
                self.lr /= 1.5
                self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        elif kl.data[0] < self.kl_targ * self.beta_adj_thres[0]:
            if self.beta_lower < self.beta:
                self.beta = self.beta / 1.5
            if self.beta < self.beta_lower * 1.5:
                self.lr *= 1.5
                self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        if self.get_info:
            info_after = self._derive_info(observes, actions, advantages, old_prob)
            return merge_before_after(info_before, info_after)


class PPO_clip_Updater(Updater):
    def __init__(self, net, probtype, epsilon, kl_target, epochs, adj_thres, clip_range, lr, get_info=True):
        self.net = net
        self.probtype = probtype
        self.clip_epsilon = epsilon
        self.kl_target = kl_target
        self.epochs = epochs
        self.get_info = get_info
        self.clip_adj_thres = adj_thres
        self.clip_upper = clip_range[1]
        self.clip_lower = clip_range[0]
        self.lr = lr
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def _derive_info(self, observations, actions, advantages, fixed_dist, fixed_prob):
        new_prob = self.net(observations)
        new_p = self.probtype.likelihood(actions, new_prob)
        prob_ratio = new_p / fixed_prob
        cliped_ratio = torch.clamp(prob_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        surr = -prob_ratio * advantages
        cliped_surr = -cliped_ratio * advantages
        clip_loss = torch.cat([surr, cliped_surr], 1).max(1)[0].mean()
        kl = self.probtype.kl(fixed_dist, new_prob).mean()

        losses = {"surr": surr.mean().data[0], "clip_surr": clip_loss.data[0], "kl": kl.data[0],
                  "ent": self.probtype.entropy(new_prob).data.mean(), 'clip_epsilon': self.clip_epsilon, 'lr': self.lr}
        return losses

    def __call__(self, path):
        observes = turn_into_cuda(path["observation"])
        actions = turn_into_cuda(path["action"])
        advantages = turn_into_cuda(path["advantage"])
        old_prob = self.net(observes).detach()
        fixed_prob = self.probtype.likelihood(actions, old_prob).detach()

        if self.get_info:
            info_before = self._derive_info(observes, actions, advantages, old_prob, fixed_prob)

        for e in range(self.epochs):
            new_prob = self.net(observes)
            new_p = self.probtype.likelihood(actions, new_prob)
            prob_ratio = new_p / fixed_prob
            cliped_ratio = torch.clamp(prob_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            surr = -prob_ratio * advantages
            cliped_surr = -cliped_ratio * advantages
            clip_loss = torch.cat([surr, cliped_surr], 1).max(1)[0].mean()

            self.net.zero_grad()
            clip_loss.backward()
            self.optimizer.step()

            prob = self.net(observes)
            kl = self.probtype.kl(old_prob, prob).mean()
            if kl.data[0] > 4 * self.kl_target:
                break

        if kl.data[0] > self.kl_target * self.clip_adj_thres[1]:
            if self.clip_lower < self.clip_epsilon:
                self.clip_epsilon = self.clip_epsilon / 1.2
            if self.clip_epsilon < self.clip_lower * 1.2:
                self.lr /= 1.5
                self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        elif kl.data[0] < self.kl_target * self.clip_adj_thres[0]:
            if self.clip_upper > self.clip_epsilon:
                self.clip_epsilon = self.clip_epsilon * 1.2
            if self.clip_epsilon > self.clip_upper / 1.2:
                self.lr *= 1.5
                self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        if self.get_info:
            info_after = self._derive_info(observes, actions, advantages, old_prob, fixed_prob)
            return merge_before_after(info_before, info_after)


# ================================================================
# Adam Optimizer
# ================================================================
class Adam_Optimizer(Optimizer):
    def __init__(self, net, lr, epochs, batch_size, get_data=True):
        self.net = net
        self.optimizer = optim.Adam(self.net.parameters(), lr)
        self.epochs = epochs
        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.get_data = get_data
        self.default_batch_size = batch_size

    def _derive_info(self, observations, y_targ):
        y_pred = self.net(observations)
        explained_var = 1 - torch.var(y_targ - y_pred) / torch.var(y_targ)
        loss = (y_targ - y_pred).pow(2).mean()
        info = {'explained_var': explained_var.data[0], 'loss': loss.data[0]}
        return info

    def __call__(self, path):
        observations = turn_into_cuda(path["observation"])
        y_targ = turn_into_cuda(path["return"])

        if self.get_data:
            info_before = self._derive_info(observations, y_targ)

        num_batches = max(observations.size()[0] // self.default_batch_size, 1)
        batch_size = observations.size()[0] // num_batches

        if self.replay_buffer_x is None:
            x_train, y_train = observations, y_targ
        else:
            x_train = torch.cat([observations, self.replay_buffer_x], dim=0)
            y_train = torch.cat([y_targ, self.replay_buffer_y], dim=0)
        self.replay_buffer_x = observations
        self.replay_buffer_y = y_targ

        for e in range(self.epochs):
            sortinds = np.random.permutation(observations.size()[0])
            sortinds = turn_into_cuda(np_to_var(sortinds).long())
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                obs_ph = x_train.index_select(0, sortinds[start:end])
                val_ph = y_train.index_select(0, sortinds[start:end])
                out = self.net(obs_ph)
                loss = (out - val_ph).pow(2).mean()
                self.net.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.get_data:
            info_after = self._derive_info(observations, y_targ)
            return merge_before_after(info_before, info_after), {}
        else:
            return {}, {}
