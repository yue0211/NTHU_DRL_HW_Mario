from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import gym_super_mario_bros

from gym import Wrapper, ObservationWrapper
from gym.spaces import Box
from collections import deque
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.optim import Adam
from random import random, randrange
import math
from torch.autograd import Variable
from torch import FloatTensor, LongTensor

from os.path import join
import gym

PRETRAINED_MODELS = '112062682_hw2_data'
environment = 'SuperMarioBros-v0'

# 检查是否有CUDA设备可用，然后选择设备
device = torch.device("cpu")

# Dueling 結構
class CNNDQN(torch.nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CNNDQN, self).__init__()
        self._input_shape = input_shape
        self._num_actions = num_actions

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU()
        ).to(device)  

        # 直接计算 feature_size 并初始化网络
        self._feature_size = self._compute_feature_size()

        self.advantage_stream = torch.nn.Sequential(
            torch.nn.Linear(self._feature_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self._num_actions)
        ).to(device)  

        self.value_stream = torch.nn.Sequential(
            torch.nn.Linear(self._feature_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        ).to(device)  

    def _compute_feature_size(self):
        # 使用一个临时的零张量来确定特征层输出的大小
        with torch.no_grad():  # 使用不需要梯度的上下文以节省内存
            return self.features(torch.zeros(1, *self._input_shape).to(device)).view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        advantage = self.advantage_stream(x)
        value = self.value_stream(x)
        q_value = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_value

    def act(self, state, epsilon):
        if random() > epsilon:
            state = torch.FloatTensor(np.float32(state)).unsqueeze(0).to(device)
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = randrange(self._num_actions)
        return action


def Observation(observation):
    # 此处代码不变，依旧将图像处理为单通道灰度图，然后扩展到4个通道
    # 确保最后返回的形状为 [4, 84, 84]
    frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    frame_stack = np.stack([frame]*4, axis=0)  # 使用np.stack确保通道数为4
    return frame_stack



class Agent:
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.net = CNNDQN((4, 84, 84), self.action_space.n).to(device)
        self.net.load_state_dict(torch.load(join(PRETRAINED_MODELS, '%s.dat' % environment), map_location=device))
        self.net.eval()  # 确保模型处于评估模式

    def act(self, observation):
        observation_space = Observation(observation)
        observation_space = np.expand_dims(observation_space, axis=0)  # 添加批量维度，形状变为 [1, 4, 84, 84]
        state_v = torch.tensor(observation_space, dtype=torch.float32).to(device)
        q_vals = self.net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        return action



# env = gym_super_mario_bros.make('SuperMarioBros-v0')
# env = JoypadSpace(env, COMPLEX_MOVEMENT)

# agent = Agent()


# total_reward = 0
# reward = 0
# done = False
# obs = env.reset()
# while not done:
#     env.render()
#     action = agent.act(obs)
#     obs, reward, done, info = env.step(action)
#     total_reward += reward
#     env.render()

# print("score: ", total_reward)

# env.close()