from train import *
from models.net_builder import *
from basic_utils.env_wrapper import Vec_env_wrapper
from models.agents import *
from basic_utils.options import *
import argparse


def train_TRPO(env='HalfCheetahBulletEnv-v0', load_model=False, render=False, save_every=None, gamma=0.99, lam=0.98):
    # set seed
    torch.manual_seed(2)

    # set environment
    env = Vec_env_wrapper(name=env, consec_frames=4, running_stat=False)
    ob_space = env.observation_space

    probtype = DiagGauss(env.action_space)

    # set neural network
    pol_net = MLPs_pol(ob_space, net_topology_pol_vec, probtype.output_layers)
    v_net = MLPs_v(ob_space, net_topology_v_vec)
    if use_cuda:
        pol_net.cuda()
        v_net.cuda()

    # set agent
    agent = TRPO_Agent(pol_net=pol_net,
                       v_net=v_net,
                       probtype=probtype,
                       lr_optimizer=1e-3,
                       epochs_v=10,
                       cg_iters=10,
                       max_kl=0.01,
                       batch_size=256,
                       cg_damping=1e-3,
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
    parser = argparse.ArgumentParser(description='')

    # arguments represent
    parser.add_argument('--env', default='HalfCheetahBulletEnv-v0', type=str, help='open-ai gym environment')

    args = parser.parse_args()
    
    train_TRPO(env=args.env)
