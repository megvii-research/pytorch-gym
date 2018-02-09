from basic_utils.layers import ConcatFixedStd, Add_One, Softplus
from basic_utils.utils import *


class StochPolicy:
    def __init__(self, net, probtype, updater):
        self.net = net
        self.probtype = probtype
        self.updater = updater

    def act(self, ob):
        ob = turn_into_cuda(np_to_var(ob))
        prob = self.net(ob).data.cpu().numpy()
        return self.probtype.sample(prob)

    def update(self, *args):
        stats = self.updater(*args)
        return stats

    def save_model(self, name):
        torch.save(self.net, name + "_policy.pkl")

    def load_model(self, name):
        net = torch.load(name + "_policy.pkl")
        self.net.load_state_dict(net.state_dict())


# ================================================================
# Abstract Class of Probtype
# ================================================================
class Probtype:
    """
    This is the abstract class of probtype.
    """
    def likelihood(self, a, prob):
        """
        Output the likelihood of an action given the parameters of the probability.

        Args:
            a: the action
            prob: the parameters of the probability

        Return:
            likelihood: the likelihood of the action
        """
        raise NotImplementedError

    def loglikelihood(self, a, prob):
        """
        Output the log likelihood of an action given the parameters of the probability.

        Args:
            a: the action
            prob: the parameters of the probability

        Return:
            log_likelihood: the log likelihood of the action
        """
        raise NotImplementedError

    def kl(self, prob0, prob1):
        """
        Output the kl divergence of two given distributions

        Args:
            prob0: the parameter of the first distribution
            prob1: the parameter of the second distribution

        Return:
            kl: the kl divergence between the two distributions
        """
        raise NotImplementedError

    def entropy(self, prob0):
        """
        Output the entropy of one given distribution

        Args:
            prob0: the parameter of the distribution

        Return:
            entropy: the entropy of the distribution
        """
        raise NotImplementedError

    def sample(self, prob):
        """
        Sample action from the given distribution.

        Args:
            prob: the parameter of the distribution

        Return:
            action: the sampled action
        """
        raise NotImplementedError

    def maxprob(self, prob):
        """
        Sample action with the maximum likelihood.

        Args:
            prob: the parameter of the distribution

        Return:
            action: the sampled action
        """
        raise NotImplementedError

    def output_layers(self, oshp):
        """
        Set the output layer needed for the distribution.

        Args:
            oshp: the input shape

        Return:
            layer: the corresponding layer
        """
        raise NotImplementedError

    def process_act(self, a):
        """
        Optional action processer.
        Args:
            a: the action to be processed

        Return:
            processed_action: the processed action
        """
        return a


class Categorical(Probtype):
    """
    The multinomial distribution for discrete action space. It gives
     a vector representing the probability for selecting each action.
    """
    def __init__(self, ac_space):
        self.n = ac_space.n

    def likelihood(self, a, prob):
        return prob.gather(1, a.long())

    def loglikelihood(self, a, prob):
        return self.likelihood(a, prob).log()

    def kl(self, prob0, prob1):
        return (prob0 * torch.log(prob0 / prob1)).sum(dim=1)

    def entropy(self, prob0):
        return - (prob0 * prob0.log()).sum(dim=1)

    def sample(self, prob):
        assert prob.ndim == 2
        N = prob.shape[0]
        csprob_nk = np.cumsum(prob, axis=1)
        return np.argmax(csprob_nk > np.random.rand(N, 1), axis=1)

    def maxprob(self, prob):
        return prob.argmax(axis=1)

    def output_layers(self, oshp):
        return [nn.Linear(oshp, self.n), nn.Softmax()]


class DiagGauss(Probtype):
    """
    The diagonal Gauss distribution for continuous action space.
    It models the distribution of the action as independent Gaussian distribution.
    """
    def __init__(self, ac_space):
        self.d = ac_space.shape[0]

    def loglikelihood(self, a, prob):
        mean0 = prob[:, :self.d]
        std0 = prob[:, self.d:]
        return - 0.5 * (((a - mean0) / std0).pow(2)).sum(dim=1, keepdim=True) - 0.5 * np.log(
            2.0 * np.pi) * self.d - std0.log().sum(dim=1, keepdim=True)

    def likelihood(self, a, prob):
        return self.loglikelihood(a, prob).exp()

    def kl(self, prob0, prob1):
        mean0 = prob0[:, :self.d]
        std0 = prob0[:, self.d:]
        mean1 = prob1[:, :self.d]
        std1 = prob1[:, self.d:]
        return ((std1 / std0).log()).sum(dim=1) + (
            (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2))).sum(dim=1) - 0.5 * self.d

    def entropy(self, prob):
        std_nd = prob[:, self.d:]
        return std_nd.log().sum(dim=1) + .5 * np.log(2 * np.pi * np.e) * self.d

    def sample(self, prob):
        mean_nd = prob[:, :self.d]
        std_nd = prob[:, self.d:]
        return np.random.randn(prob.shape[0], self.d) * std_nd + mean_nd

    def maxprob(self, prob):
        return prob[:, :self.d]

    def output_layers(self, oshp):
        return [nn.Linear(oshp, self.d), ConcatFixedStd(self.d)]
