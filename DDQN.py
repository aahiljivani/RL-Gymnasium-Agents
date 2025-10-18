import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'truncated'))

class DQN(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super().__init__()
        self.layer1 = nn.Linear(state_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, action_size)
        
        # He initialization to stabilize gradients since we use ReLU activation functions
        # set the weights
        nn.init.kaiming_normal_(self.layer1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer3.weight, nonlinearity='relu')
        # set the biases to 0
        nn.init.constant_(self.layer1.bias, 0.0)
        nn.init.constant_(self.layer2.bias, 0.0)
        nn.init.constant_(self.layer3.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class DQN_Agent():
    def __init__(self):
        self.env = gym.make("CartPole-v1", render_mode="human")
        # Hyperparameters for CartPole 
        self.batch_size = 64
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995  # Faster decay for earlier exploitation
        self.gamma = 0.99
        self.learning_rate = 3e-4  # Lower for stability
        self.epsilon_min = 0.01
        self.rewards = []
        # DQN inputs
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.hidden_size = 64  # Increased for better capacity

        self.policy = DQN(self.state_size, self.hidden_size, self.action_size)
        self.target = DQN(self.state_size, self.hidden_size, self.action_size)
        self.target.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.loss = nn.MSELoss()

    def epsilon_greedy(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                # Unsqueeze for batch dimension (state is (4,) -> (1, 4))
                q_values = self.policy(state.unsqueeze(0))
                return q_values.argmax().item()

    def sync(self):
        '''Sync the policy model weights and biases with the target model weights and biases'''
        self.target.load_state_dict(self.policy.state_dict())

    def random_sample(self):
        return random.sample(self.memory, self.batch_size)

    def sgd(self):
        if len(self.memory) < 1000:  # Warmup buffer
            return None
        else:
            # Sample a batch from memory
            sample_batch = self.random_sample()
            # Format batches
            batch = Transition(*zip(*sample_batch))

            # Stack tensors (states/next_states are (4,) -> stack to (batch, 4))
            states = torch.stack(batch.state)
            actions = torch.LongTensor(batch.action).unsqueeze(1)
            next_states = torch.stack(batch.next_state)
            rewards = torch.tensor(batch.reward, dtype=torch.float32)
            dones = torch.tensor(batch.done, dtype=torch.bool)
            truncs = torch.tensor(batch.truncated, dtype=torch.bool)
            terminals = dones | truncs  # Combined terminal mask

            # Calculate Q values in batch format
            q_values = self.policy(states).gather(1, actions).squeeze()

            with torch.no_grad():
                next_q_values = self.target(next_states).max(1)[0]
                next_q_values[terminals] = 0  # Zero for terminal states
                target_q_values = rewards + self.gamma * next_q_values

            # Compute loss and update
            loss = self.loss(q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item()

    def train(self, episodes=1000):
        step = 0

        for episode in range(episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32)  # Flat (4,)
            done = False
            truncated = False
            total_reward = 0

            while not (done or truncated):
                action = self.epsilon_greedy(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32)  # Flat (4,)
                # Save transition (states are flat tensors)
                transition = Transition(state, action, reward, next_state, done, truncated)
                self.memory.append(transition)
                state = next_state
                total_reward += reward

                loss_val = self.sgd()
                step += 1

            # Sync target network (every 1000 steps + every 10 episodes for stability)
            if step % 1000 == 0 or episode % 10 == 0:
                self.sync()

            # Epsilon decay per episode
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            self.rewards.append(total_reward)

            # Print progress every 50 episodes
            if episode % 50 == 0:
                avg_reward = np.mean(self.rewards[-50:]) if len(self.rewards) >= 50 else np.mean(self.rewards)
                print(f"Episode {episode}/{episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")

        return self.rewards

if __name__ == "__main__":
    agent = DQN_Agent()
    rewards = agent.train(episodes=1000)
    print("Training complete!")
