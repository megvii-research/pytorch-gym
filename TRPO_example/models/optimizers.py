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
# Trust Region Policy Optimization Updater
# ================================================================
class TRPO_Updater(Updater):
    def __init__(self, net, probtype, max_kl, cg_damping, cg_iters, get_info):
        self.net = net
        self.probtype = probtype
        self.max_kl = max_kl
        self.cg_damping = cg_damping
        self.cg_iters = cg_iters
        self.get_info = get_info

    def conjugate_gradients(self, b, nsteps, residual_tol=1e-10):
        x = torch.zeros(b.size())
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            Avp = self.Fvp(p)
            alpha = rdotr / torch.dot(p, Avp)
            x += alpha * p
            r -= alpha * Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x

    def linesearch(self, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
        fval = self.get_loss().data
        for (_n_backtracks, stepfrac) in enumerate(.5 ** np.arange(max_backtracks)):
            xnew = x + stepfrac * fullstep
            set_flat_params_to(self.net, xnew)
            newfval = self.get_loss().data
            actual_improve = fval - newfval
            expected_improve = expected_improve_rate * stepfrac
            ratio = actual_improve / expected_improve

            if ratio[0] > accept_ratio and actual_improve[0] > 0:
                return True, xnew
        return False, x

    def Fvp(self, v):
        kl = self.probtype.kl(self.fixed_dist, self.net(self.observations)).mean()
        grads = torch.autograd.grad(kl, self.net.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
        kl_v = (flat_grad_kl * Variable(v)).sum()
        grads = torch.autograd.grad(kl_v, self.net.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data.cpu()

        return flat_grad_grad_kl + v * self.cg_damping

    def get_loss(self):
        prob = self.net(self.observations)
        prob = self.probtype.likelihood(self.actions, prob)
        action_loss = -self.advantages * prob / self.fixed_prob
        return action_loss.mean()

    def _derive_info(self, observations, actions, advantages, fixed_dist, fixed_prob):
        new_prob = self.net(observations)
        new_p = self.probtype.likelihood(actions, new_prob)
        prob_ratio = new_p / fixed_prob
        surr = torch.mean(prob_ratio * advantages)
        losses = {"surr": -surr.data[0], "kl": self.probtype.kl(fixed_dist, new_prob).data.mean(),
                  "ent": self.probtype.entropy(new_prob).data.mean()}

        return losses

    def __call__(self, path):
        self.observations = turn_into_cuda(path["observation"])
        self.actions = turn_into_cuda(path["action"])
        self.advantages = turn_into_cuda(path["advantage"])
        self.fixed_dist = self.net(self.observations).detach()
        self.fixed_prob = self.probtype.likelihood(self.actions, self.fixed_dist).detach()

        if self.get_info:
            info_before = self._derive_info(self.observations, self.actions, self.advantages, self.fixed_dist,
                                            self.fixed_prob)

        loss = self.get_loss()
        grads = torch.autograd.grad(loss, self.net.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).data.cpu()

        stepdir = self.conjugate_gradients(-loss_grad, self.cg_iters)
        shs = 0.5 * (stepdir * self.Fvp(stepdir)).sum(0, keepdim=True)
        lm = torch.sqrt(shs / self.max_kl)
        fullstep = turn_into_cuda(stepdir / lm[0])
        neggdotstepdir = turn_into_cuda((-loss_grad * stepdir).sum(0, keepdim=True))
        prev_params = get_flat_params_from(self.net)
        success, new_params = self.linesearch(prev_params, fullstep, neggdotstepdir / lm[0])
        set_flat_params_to(self.net, new_params)

        if self.get_info:
            info_after = self._derive_info(self.observations, self.actions, self.advantages, self.fixed_dist,
                                           self.fixed_prob)
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
