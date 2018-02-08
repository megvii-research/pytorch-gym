from train import *
from models.net_builder import *
from basic_utils.env_wrapper import Vec_env_wrapper
from models.agents import *
from basic_utils.options import *


def train_CartPole_adapted_PPO(load_model=False, render=False, save_every=None, gamma=0.99, lam=0.98):
    # set seed
    torch.manual_seed(2)

    # set environment
    env = Vec_env_wrapper(name='HalfCheetahBulletEnv-v0', consec_frames=1, running_stat=True)
    ob_space = env.observation_space

    probtype = DiagGauss(env.action_space)

    # set neural network
    pol_net = MLPs_pol(ob_space, net_topology_pol_vec, probtype.output_layers)
    v_net = MLPs_v(ob_space, net_topology_v_vec)
    if use_cuda:
        pol_net.cuda()
        v_net.cuda()

    # set agent
    agent = PPO_adapted_Agent(pol_net,
                              v_net,
                              probtype,
                              epochs_v=10,
                              epochs_p=10,
                              kl_target=0.01,
                              lr_updater=9e-4,
                              lr_optimizer=1e-3,
                              batch_size=256,
                              adj_thres=(0.5, 2.0),
                              beta=1.0,
                              beta_range=(1 / 35.0, 35.0),
                              kl_cutoff_coeff=50.0,
                              get_info=True)
    if load_model:
        agent.load_model("./save_model/" + env.name + "_" + agent.name)

    # set data processor
    single_processors = [
        Scale_Reward(1 - gamma),
        Calculate_Return(gamma),
        Predict_Value(agent.baseline),
        Calculate_Generalized_Advantage(gamma, lam),
        Extract_Item_By_Name(["observation", "action", "advantage", "return"]),
        Concatenate_Paths()
    ]
    processor = Ensemble(single_processors)

    # set data generator
    generator = Parallel_Path_Data_Generator(agent=agent,
                                             env=env,
                                             n_worker=10,
                                             path_num=1,
                                             action_repeat=1,
                                             render=render)
    # set trainer
    t = Path_Trainer(agent,
                     env,
                     data_generator=generator,
                     data_processor=processor,
                     save_every=save_every,
                     print_every=10)
    t.train()


def train_CartPole_clip_PPO(load_model=False, render=False, save_every=None, gamma=0.99, lam=0.98):
    # set seed
    torch.manual_seed(2)

    # set environment
    env = Vec_env_wrapper(name='HalfCheetahBulletEnv-v0', consec_frames=1, running_stat=True)
    ob_space = env.observation_space

    probtype = DiagGauss(env.action_space)

    # set neural network
    pol_net = MLPs_pol(ob_space, net_topology_pol_vec, probtype.output_layers)
    v_net = MLPs_v(ob_space, net_topology_v_vec)
    if use_cuda:
        pol_net.cuda()
        v_net.cuda()

    agent = PPO_clip_Agent(pol_net,
                           v_net,
                           probtype,
                           epochs_v=10,
                           epochs_p=10,
                           kl_target=0.01,
                           lr_updater=9e-4,
                           lr_optimizer=1e-3,
                           batch_size=256,
                           adj_thres=(0.5, 2.0),
                           clip_range=(0.05, 0.3),
                           epsilon=0.2,
                           get_info=True)
    if load_model:
        agent.load_model("./save_model/" + env.name + "_" + agent.name)

    # set data processor
    single_processors = [
        Scale_Reward(1 - gamma),
        Calculate_Return(gamma),
        Predict_Value(agent.baseline),
        Calculate_Generalized_Advantage(gamma, lam),
        Extract_Item_By_Name(["observation", "action", "advantage", "return"]),
        Concatenate_Paths()
    ]
    processor = Ensemble(single_processors)

    # set data generator
    generator = Parallel_Path_Data_Generator(agent=agent,
                                             env=env,
                                             n_worker=10,
                                             path_num=1,
                                             action_repeat=1,
                                             render=render)

    # set trainer
    t = Path_Trainer(agent,
                     env,
                     data_generator=generator,
                     data_processor=processor,
                     save_every=save_every,
                     print_every=10)
    t.train()


if __name__ == '__main__':
    # train_CartPole_adapted_PPO()
    train_CartPole_clip_PPO()
