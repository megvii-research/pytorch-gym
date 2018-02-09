# Qin Yongliang
# start multiple environment in parallel

from llll import PythonInstance

slave_code = '''
from llll import sbc
sbc = sbc()
import numpy
import gym

# import everything relevant here

envname = sbc.recv()
env = gym.make(envname)

print('environment instantiated:', envname)

while True:
    obj = sbc.recv() # tuple pair of funcname and arg
    f = obj[0]
    arg = obj[1]

    if f == 'reset':
        obs = env.reset()
        sbc.send(obs)

    elif f == 'step':
        ordi = env.step(arg)
        sbc.send(ordi)

    else:
        print('unsupported:',f,arg)
        break

'''

class RemoteEnv:
    def __init__(self,envname):
        self.pi = PythonInstance(slave_code)
        self.pi.send(envname)
    def reset(self,):
        self.pi.send(('reset', None))
        return self.pi.recv()
    def step(self,action):
        self.pi.send(('step', action))
        return self.pi.recv()

if __name__ == '__main__':
    import numpy,gym
    envname = 'CartPole-v1'
    local_env = gym.make(envname)

    # create 16 envs in parallel
    remote_envs = [RemoteEnv(envname) for i in range(16)]

    # step 10 steps in parallel on 16 envs
    for e in remote_envs: e.reset()
    for j in range(10):
        for e in remote_envs:
            o,r,d,i = e.step(local_env.action_space.sample())
            print(o)
