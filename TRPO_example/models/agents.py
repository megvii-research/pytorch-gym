from basic_utils.options import *
import numpy as np
from models.net_builder import MLPs_pol, MLPs_v
from models.optimizers import *
from models.policies import StochPolicy
from models.baselines import *
from models.data_processor import *


# ================================================================
# Abstract Class
# ================================================================
class BasicAgent:
    """
    This is the abstract class of the agent.
    """

    def act(self, ob_no):
        """
        Get the action given the observation.

        Args:
            ob_no: the observation

        Return:
            the corresponding action
        """
        raise NotImplementedError

    def update(self, paths):
        """
        Update the weights of the network.

        Args:
            paths: a dict containing the information for updating

        Return:
            information of the updating process, extra information
        """
        raise NotImplementedError

    def get_params(self):
        """
        Get the parameters of the agent.

        Return:
            the state dict of the agent
        """
        raise NotImplementedError

    def set_params(self, state_dicts):
        """
        Set the parameters to the agent.

        Args:
            state_dicts: the parameters to be set
        """
        raise NotImplementedError

    def save_model(self, name):
        """
        Save the model.
        """
        raise NotImplementedError

    def load_model(self, name):
        """
        Load the model.
        """
        raise NotImplementedError


# ================================================================
# Policy Based Agent
# ================================================================
class Policy_Based_Agent(BasicAgent):
    def __init__(self, policy, baseline):
        self.policy = policy
        self.baseline = baseline

    def act(self, observation):
        return self.policy.act(observation)

    def process_act(self, action):
        return self.policy.probtype.process_act(action)

    def update(self, processed_path):
        vf_name, vf_stats, info = self.baseline.fit(processed_path)
        pol_stats = self.policy.update(processed_path)
        return [(vf_name, vf_stats), ("pol", pol_stats)], info

    def save_model(self, name):
        self.policy.save_model(name)
        self.baseline.save_model(name)

    def load_model(self, name):
        self.policy.load_model(name)
        self.baseline.load_model(name)

    def get_params(self):
        return self.policy.net.state_dict(), self.baseline.net.state_dict()

    def set_params(self, state_dicts):
        self.policy.net.load_state_dict(state_dicts[0])
        self.baseline.net.load_state_dict(state_dicts[1])


# ================================================================
# Trust Region Policy Optimization
# ================================================================
class TRPO_Agent(Policy_Based_Agent):
    name = 'TRPO_Agent'

    def __init__(self,
                 pol_net,
                 v_net,
                 probtype,
                 lr_optimizer=1e-3,
                 epochs_v=10,
                 cg_iters=10,
                 max_kl=0.003,
                 batch_size=256,
                 cg_damping=1e-3,
                 get_info=True):
        updater = TRPO_Updater(net=pol_net,
                               probtype=probtype,
                               cg_damping=cg_damping,
                               cg_iters=cg_iters,
                               max_kl=max_kl,
                               get_info=get_info)

        optimizer = Adam_Optimizer(net=v_net,
                                   batch_size=batch_size,
                                   epochs=epochs_v,
                                   lr=lr_optimizer)

        policy = StochPolicy(net=pol_net, probtype=probtype, updater=updater)
        baseline = ValueFunction(net=v_net, optimizer=optimizer)

        Policy_Based_Agent.__init__(self, baseline=baseline, policy=policy)
