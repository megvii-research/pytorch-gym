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

image_width = 16
circle = cv2.imread('circle.png').astype('uint8') #saves space
circle = cv2.resize(circle, dsize=(image_width, image_width), interpolation=cv2.INTER_CUBIC)

def load_random_image():
    # load a random image and return. byref is preferred.
    # here we return the same image everytime.
    bias = np.random.uniform(size=(1,1,3))*0.3
    gain = np.random.uniform(size=(1,1,3))*0.7 + 0.7
    randomized_lena = np.clip(circle * gain + bias, 0, 255).astype('uint8')
    return randomized_lena

# Environment Specification
# observation: tuple(image, image)
# action: Box(3) clamped to [0,1]

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

class CanvasEnv:
    def __init__(self):
        self.action_dims = ad = 2 * image_width
        self.action_space = Box(np.array([-1.] * ad), np.array([1.] * ad))
        self.observation_space = Box(np.zeros([image_width, image_width, 2]), np.ones([image_width, image_width, 2]))
        self.target_drawn = False
        self.target = circle
        self.canvas = np.zeros(shape=self.target.shape, dtype='uint8') + 255
        self.stepnum = 0

    def reset(self):
        self.stepnum = 0
        # target should be a 3-channel colored image of shape (H,W,3) in uint8
        self.target_drawn = False
        self.canvas = np.zeros(shape=self.target.shape, dtype='uint8') + 255
        self.lastdiff = self.diff()
        self.rewardscale = self.lastdiff
        self.height, self.width, self.depth = self.canvas.shape
        r = self.height // 2
        white = (255, 255, 255)
        # cv2.circle(self.canvas, (r, r), r, white, -1)
        return self.observation()
    
    def diff(self):
        # calculate dDifference between two image. you can use different metrics to encourage different characteristic
        p = 1 - (self.target[:, :, 0].astype('uint8') / 255)
        q = 1 - (self.canvas[:, :, 0].astype('uint8') / 255)
        return np.sum(np.logical_xor(p, q).astype('uint8')) / image_width / image_width
    
    def observation(self):
        p = self.target[:, :, 0].astype('uint8') / 255
        q = self.canvas[:, :, 0].astype('uint8') / 255
        ob = np.stack(np.array([p, q]), axis=2)
        return ob # np.array(self.target), np.array(self.canvas)

    def draw(self, x, y, r):
        if(r == 1):
            self.canvas[y, x] = (0, 0, 0)
            return
        x = x * 16
        y = y * 16
        r = r * 16
        cv2.rectangle(
            self.canvas,
            (int(x),int(y)),
            (int(x+r),int(y+r)),
            (0,0,0),
            -1,
            cv2.LINE_8,
            4
        )

    def step(self, action):
        # x = np.argmax(action[:image_width])
        # r = (np.argmax(action[image_width:]) - 1)
        # pic = self.canvas[:, :, 0]
        # if (r != -1):
        #    r = 2 ** r
        #    for i in range(image_width):
        #        if(np.sum(pic[i, x : x + r + 1])):
        #            self.draw(x, i, r)
        #            break
        x = (action[:image_width] + 1) / 2.
        y = (action[image_width:] + 1) / 2.
        x = softmax(x)
        y = softmax(y)
        grey = x * y.reshape(image_width, 1)
        grey = grey.reshape((image_width, image_width, 1))
        grey = (grey * (255, 255, 255) * image_width).astype('uint8')
        grey = np.minimum(grey, self.canvas)
        self.canvas -= grey
        diff = self.diff()
        reward = (self.lastdiff - diff) / self.rewardscale # reward is positive if diff increased
        self.lastdiff = diff
        self.stepnum += 1
        ob = self.observation()
        # self.canvas = np.stack(np.rot90(self.canvas))
        # self.target = np.stack(np.rot90(self.target))
        return ob, reward, (self.stepnum >= 10), None # o,r,d,i
    
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
        time.sleep(0.4)
        tot_reward += reward
        if step % 20 == 0:
            time.sleep(2)
            cv2.imwrite(str(step) + '.png', env.canvas, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            print('step {} reward {}'.format(step, tot_reward))
            ob = env.reset()
            tot_reward = 0
