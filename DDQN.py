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

        self.transition = NamedTuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'truncated'))
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
        self.target.load_state_dict(self.policy.state_dict()) # load the same weights from the policy onto the network
        self.step = 0 # global step counter
        self.rewards = []


    def epsilon_greedy(self, state):
        if np.random.rand < self.epsilon:
            return self.env.action_space.sample()

        else:
            with torch.no_grad():
                q_values = self.policy(state)
                
                return np.argmax(q_values.unsqueeze)


    def train_agent(self, episodes):
        for _ in range(episodes):
            state = self.env.reset()
            state = torch.tensor(state).unsqueeze(0) # reformatted for forward pass through NN
            while not (done or truncated):

                action = self.epsilon_greedy(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_state = torch.tensor(next_state).unsqueeze(0) # reformatted for batch pass
                self.transition(state, action, reward, next_state) # format into tuple
                # save transition to memory
                self.replay_buffer.append(self.transition)





            


