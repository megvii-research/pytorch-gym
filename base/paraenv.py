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
        def future():
            return self.pi.recv()
        return future
    def step(self,action):
        self.pi.send(('step', action))
        def future():
            return self.pi.recv()
        return future

if __name__ == '__main__':
    import numpy,gym
    envname = 'CartPole-v1'
    local_env = gym.make(envname)

    # create 16 envs in parallel
    remote_envs = [RemoteEnv(envname) for i in range(16)]

    # step 20 steps in parallel on 16 envs

    # reset all envs and obtain first observation
    futures = []
    for e in remote_envs:
        future = e.reset()
        futures.append(future)

    for future in futures:
        obs = future()
        print(obs)

    # for 20 steps:
    for j in range(20):

        # step all 16 envs simultaneously
        futures = []
        for e in remote_envs:
            future = e.step(local_env.action_space.sample())
            futures.append(future)

        # collect results simultaneously
        for future in futures:
            o,r,d,i = future()
            print(o)
