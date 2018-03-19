# the reinforcement learning painting environment.
# run this file directly to test functionality.
# Qin Yongliang 20171029

import time 
import numpy as np

# OpenCV3.x opencv-python
import cv2

# provide shorthand for auto-scaled imshow(). the library is located at github/ctmakro/cv2tools
from cv2tools import vis,filt

# we provide a Gym-like interface. instead of inheriting directly, we steal only the Box descriptor.
import gym
from gym.spaces import Box

circle = cv2.imread('circle.png').astype('uint8') #saves space
circle = cv2.resize(circle, dsize=(84, 84), interpolation=cv2.INTER_CUBIC)

def load_random_image():
    # load a random image and return. byref is preferred.
    # here we return the same image everytime.
    bias = np.random.uniform(size=(1,1,3))*0.3
    gain = np.random.uniform(size=(1,1,3))*0.7 + 0.7
    randomized_lena = np.clip(circle * gain + bias, 0, 255).astype('uint8')
    return randomized_lena

def ang2point(x): # [0, 1]
    r = 0.5
    alpha = x * np.pi * 2 - np.pi
    px = r + r * np.sin(alpha)
    py = r - r * np.cos(alpha)
    return px, py

# Environment Specification
# observation: tuple(image, image)
# action: Box(3) clamped to [0,1]

class CanvasEnv:
    def __init__(self):
        self.action_dims = ad = 3
        self.action_space = Box(np.array([-1.] * ad), np.array([1.] * ad))
        self.observation_space = Box(np.zeros([84, 84, 2]), np.ones([84, 84, 2]))
        self.target_drawn = False
        self.target = circle
        self.canvas = np.zeros(shape=self.target.shape, dtype='uint8') + 255
        self.stepnum = 0

    def reset(self):
        self.stepnum = 0
        # target should be a 3-channel colored image of shape (H,W,3) in uint8
        self.target_drawn = False
        self.canvas = np.zeros(shape=self.target.shape, dtype='uint8') + 255
        self.lastdiff = 0
        self.height, self.width, self.depth = self.canvas.shape
        r = self.height // 2
        white = (255, 255, 255)
        # cv2.circle(self.canvas, (r, r), r, white, -1)
        return self.observation()
    
    def diff(self):
        # calculate dDifference between two image. you can use different metrics to encourage different characteristic
        p = 255 - self.target[:, :, 0]
        q = 255 - self.canvas[:, :, 0]        
        return np.sum(np.logical_and(p, q).astype(np.float32)) / np.sum(np.logical_or(p, q).astype(np.float32))
    
    def observation(self):
        p = self.target[:, :, 0]
        q = self.canvas[:, :, 0]
        ob = np.stack(np.array([p, q]), axis=2)
        return ob #np.array(self.target), np.array(self.canvas)

    def draw(self, x1, y1, ang):
        if np.square(x1 - 0.5) + np.square(y1 - 0.5) > 0.25:
            return
        x2, y2 = ang2point(ang)
        r = self.height / 2
        sheight, swidth = r * 32, r * 32 # leftshift bits        
        cv2.line(
            self.canvas,
            (int(x1 * swidth), int(y1 * sheight)),
            (int(x2 * swidth), int(y2 * sheight)),
            (0, 0, 0), # black
            1,
            cv2.LINE_8, # antialiasing
            4 # rightshift bits
        )        
        
    def step(self, action):
        # unpack the parameters
        x1, y1, r = [(np.clip(action[i],-1.,1.) + 1.) / 2. for i in range(self.action_dims)]
        sheight, swidth = self.height * 16, self.width * 16 #
        cv2.circle(
            self.canvas,
            (int(x1*swidth),int(y1*sheight)),
            int(r*swidth/4.),
            (0,0,0),
            -1,
            cv2.LINE_AA,
            4
        )
        # calculate reward
        diff = self.diff()
        reward = diff - self.lastdiff # reward is positive if diff increased
        self.lastdiff = diff

        self.stepnum += 1
        return self.observation(), reward, (self.stepnum >= 5), None # o,r,d,i

    def render(self):
        if self.target_drawn == False:
            vis.show_autoscaled(self.target,limit=300,name='target')
            self.target_drawn = True
        vis.show_autoscaled(self.canvas,limit=300,name='canvas')

if __name__ == '__main__':
    env = CanvasEnv()
    ob = env.reset()
    tot_reward = 0.
    for step in range(2000):
        ob, reward, d, i = env.step(env.action_space.sample())
        env.render()
        tot_reward += reward
        if step % 3 == 0:
            time.sleep(1)
            cv2.imwrite(str(step) + '.png', env.canvas, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            print('step {} reward {}'.format(step, tot_reward))
            ob = env.reset()
            tot_reward = 0
