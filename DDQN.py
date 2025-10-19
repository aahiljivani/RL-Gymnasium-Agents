from typing_extensions import NamedTuple
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import gymnasium as gym
import random
import numpy as np
from collections import namedtuple, deque


class DQN(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super().__init__
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self,fc3 = nn.Linear(hidden_size, action_size)

        


    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQN_Agent():
    def __init__(self, str(env)):
        # set the environment variables
        self.env = gym.make(str(env), render_mode = "human")
        self.action_space = self.env.action_space.n
        self.state_space = self.env.observation_space.shaoe[0]
        self.hidden_layers = 64

        self.transition = NamedTuple('Transition', ('state', 'action', 'reward', 'done', 'truncated'))
        self.replay_buffer = deque(maxlen=10000) # replay buffer
        self.batch_size = 64
        self.alpha = 3e-4 # learning rate
        self.gamma = 0.99
        # set epsilon and decay parameters
        self.epsilon = 1 
        self.ep_decay = 0.995
        self.ep_min = 0.1
        # policy and target networks
        self.policy = DQN(self.state_space, self.hidden_layers, self.action_space)
        self.target = DQN(self.state_space, self.hidden_layers, self.action_space)
        self.target.load_state_dict(self.policy.state_dict())
        self.step = 0
        self.rewards = []


    def epsilon_greedy(self):
        


