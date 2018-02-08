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
# Proximal Policy Optimization
# ================================================================
class PPO_adapted_Agent(Policy_Based_Agent):
    name = 'PPO_adapted_Agent'

    def __init__(self,
                 pol_net,
                 v_net,
                 probtype,
                 epochs_v=10,
                 epochs_p=10,
                 kl_target=0.003,
                 lr_updater=9e-4,
                 lr_optimizer=1e-3,
                 batch_size=256,
                 adj_thres=(0.5, 2.0),
                 beta=1.0,
                 beta_range=(1 / 35.0, 35.0),
                 kl_cutoff_coeff=50.0,
                 get_info=True):
        updater = PPO_adapted_Updater(adj_thres=adj_thres,
                                      beta=beta,
                                      beta_range=beta_range,
                                      epochs=epochs_p,
                                      kl_cutoff_coeff=kl_cutoff_coeff,
                                      kl_target=kl_target,
                                      lr=lr_updater,
                                      net=pol_net,
                                      probtype=probtype,
                                      get_info=get_info)

        optimizer = Adam_Optimizer(net=v_net,
                                   batch_size=batch_size,
                                   epochs=epochs_v,
                                   lr=lr_optimizer)

        policy = StochPolicy(net=pol_net, probtype=probtype, updater=updater)
        baseline = ValueFunction(net=v_net, optimizer=optimizer)

        Policy_Based_Agent.__init__(self, baseline=baseline, policy=policy)


class PPO_clip_Agent(Policy_Based_Agent):
    def __init__(self,
                 pol_net,
                 v_net,
                 probtype,
                 epochs_v=10,
                 epochs_p=10,
                 kl_target=0.003,
                 lr_updater=9e-4,
                 lr_optimizer=1e-3,
                 batch_size=256,
                 adj_thres=(0.5, 2.0),
                 clip_range=(0.05, 0.3),
                 epsilon=0.2,
                 get_info=True):
        updater = PPO_clip_Updater(adj_thres=adj_thres,
                                   clip_range=clip_range,
                                   epsilon=epsilon,
                                   epochs=epochs_p,
                                   kl_target=kl_target,
                                   lr=lr_updater,
                                   net=pol_net,
                                   probtype=probtype,
                                   get_info=get_info)
        optimizer = Adam_Optimizer(net=v_net,
                                   batch_size=batch_size,
                                   epochs=epochs_v,
                                   lr=lr_optimizer)

        policy = StochPolicy(net=pol_net, probtype=probtype, updater=updater)
        baseline = ValueFunction(net=v_net, optimizer=optimizer)

        self.name = 'PPO_clip_Agent'
        Policy_Based_Agent.__init__(self, baseline=baseline, policy=policy)
