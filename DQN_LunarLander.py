import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple


Transition = namedtuple('transition', ('state', 'action', 'reward', 'next_state', 'done', 'truncated'))

class DQN(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super().__init__()
        self.layer1 = nn.Linear(state_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, action_size)

    def forward(self,x):
        
        # relu activation function layers
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class DQN_Agent():
    def __init__(self):
        self.env = gym.make("LunarLander-v3", render_mode = "human")
        self.batch_size = 32
        self.memory = deque(maxlen=5000)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.learning_rate = 1e-3
        self.epsilon_min = 0.01
        self.rewards = []
        # set DQN inputs
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.hidden_size = 128

        self.policy = DQN(self.state_size, self.hidden_size, self.action_size)
        self.target = DQN(self.state_size, self.hidden_size, self.action_size)

        self.optimizer = optim.Adam(self.policy.parameters(), lr = self.learning_rate)
        self.loss = nn.MSELoss()


    def epsilon_greedy(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample() # return random action following uniform distribution
        else:
            with torch.no_grad(): # make sure that gradients are not being calculated
                q_values = self.policy(state) # pass state through the network and get action values
                return q_values.argmax().item() # return the action with the highest action value

    def sync(self):
        ''' Sync the policy model weights and biases with the target model weights and biases'''
        self.target.load_state_dict(self.policy.state_dict())

    def random_sample(self):
        return self.random_sample(self.memory, self.batch_size)


    def sgd(self):
        if len(self.memory) <= self.batch_size:
            return None
        else:
            # sample a batch from memory
            sample_batch = random.sample(self.memory, self.batch_size)
            # format batches so that states are algined with states, actions with actions etc. 
            batch = Transition(*zip(*sample_batch))

            # Since our batches are now together we can put them into tensors
            states = torch.cat(batch.state, dim = 0)
            actions = torch.LongTensor(batch.action).unsqueeze(1)
            next_states = torch.cat(batch.next_state, dim = 0)
            rewards = torch.tensor(batch.reward, dtype = torch.float32)
            dones = torch.tensor(batch.done, dtype = torch.bool)
            truncs = torch.tensor(batch.truncated, dtype = torch.bool)

            # Now we can calculate Q values in batch format
            q_values = self.policy(states).gather(1,actions).squeeze()
            with torch.no_grad():
                next_q_values = self.target(next_states).max(1)[0] # finds the max q-value for state action pair along action dimension
                next_q_values[dones] = 0 # setting the next_q_values for the done states as zero
                next_q_values[truncs] = 0 # setting the next_q_values for the truncated states as zero
                target_q_values = rewards + self.gamma * next_q_values # compute the target q-values for the not done or truncated states

            # compute the loss between policy and target 
            loss = self.loss(q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item()


    def train(self, episodes = 1000):
        for episode in range(episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype = torch.float32).unsqueeze(0)
            done = False
            truncated = False
            total_reward = 0
            step = 0
            while not (done or truncated):
                action = self.epsilon_greedy(state)
                next_state, reward, done, truncated,_ = self.env.step(action)
                next_state = torch.tensor(next_state, dtype = torch.float32).unsqueeze(0)
                # save state transitions to named tuple for convenience
                transition = Transition(state, action, reward, next_state, done, truncated)
                # append transition to memory
                self.memory.append(transition)
                state = next_state
                total_reward += reward

                loss_val = self.sgd()
                step += 1
            # sync target network to policy network
            if episode % 10 == 0:
                self.sync()
            # epsilon decay
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            self.rewards.append(total_reward)

            # Print progress
            if episode % 50 == 0:
                avg_reward = np.mean(self.rewards[-50:]) if len(self.rewards) >= 50 else np.mean(self.rewards)
                print(f"Episode {episode}/{episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")
        
        return self.rewards



if __name__ == "__main__":
    agent = DQN_Agent()
    rewards = agent.train(episodes=1000)
    print("Training complete!")




                



            


    
 