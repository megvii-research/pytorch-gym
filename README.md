# DDPG in bullet Gym using pytorch
## Overview
This is an implementation of Deep Deterministic Policy Gradient (DDPG) in bullet Gym using PyTorch.
## Dependencies
* Python 3.6
* PyTorch 0.3.0
* openAI Gym
* pybullet

## Run
### Training:
* here is a simple example to train CartPole with high efficiency:
> $ python main.py --debug --discrete --env=CartPole-v0 --vis --tau=0.02 --discount=0.97 --warmup=1000 --window_length=2
* the results after 1 min training: 
> #240: episode_reward:129.00 steps:1107 noise:0.28 time:4.10,3.15 </br>
> #241: episode_reward:132.00 steps:1140 noise:0.24 time:2.14,3.17 </br>
> #242: episode_reward:107.00 steps:1167 noise:0.12 time:1.73,3.24 </br>
> #243: episode_reward:63.00 steps:1183 noise:0.21 time:1.00,3.21 </br>
> #244: episode_reward:117.00 steps:1213 noise:0.24 time:1.90,3.16 </br>
> #245: episode_reward:140.00 steps:1248 noise:0.14 time:2.27,3.82 </br>
> #246: episode_reward:73.00 steps:1267 noise:0.45 time:0.09,3.41 </br>
> #247: episode_reward:194.00 steps:1316 noise:0.08 time:3.19,3.21 </br>
> #248: episode_reward:306.00 steps:1393 noise:0.14 time:5.05,3.18 </br>
> #249: episode_reward:199.00 steps:1443 noise:0.03 time:3.27,3.14 </br>
> [Evaluate] Step_0001463: mean_reward:343.0 and save model

* **you can using this to understand each argument's usage:**
> $ python main.py --help

* some important arguments' explanation:
> --debug: add the argument if you want to see the episode_reward and something other information </br>
> --discrete: add the argument if the actions are discrete rather than continuous </br>
> --vis: add the argument if you want to visualize each action (but it would slow down your training speed)

## TODO
