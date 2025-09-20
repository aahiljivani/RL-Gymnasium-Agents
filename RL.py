import numpy as np
import random
import gymnasium as gym
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
import time

class MC_BlackJack_Agent:
    def __init__(self):
        self.env = gym.make("Blackjack-v1", sab=False, render_mode='rgb_array')

    
    
    def train_agent(self, gamma=1, training_episodes=500_000):
        Q_dict = defaultdict(float)
        freq = defaultdict(int)
        print(f'--- Started Training the Monte Carlo Agent on {training_episodes} games ---')
        
        for episode in range(training_episodes):
            terminated = False
            state, _ = self.env.reset()
            q_list = []
            while not terminated:
                action = self.env.action_space.sample()
                next_state, reward, terminated, _, _ = self.env.step(action)
                q_list.append((state, int(action), reward))
                state = next_state

            G = 0
            visited = set()
            for state_act, action, reward in reversed(q_list):
                G = reward + gamma * G
                if (state_act, action) not in visited:
                    freq[(state_act, action)] += 1
                    Q_dict[(state_act, action)] += (G - Q_dict[(state_act, action)]) / freq[(state_act, action)]
                    visited.add((state_act, action))

        print("\n--- Training Finished ---")
        return Q_dict
    
    def evaluate_agent(self, Q_dict, TEST_EPISODES=10):
        print("\n--- Starting Visual Evaluation ---")
        wins, losses, draws = 0, 0, 0  # initialize tracking variables

        for _ in range(TEST_EPISODES):
            state, _ = self.env.reset()
            terminated = False

            while not terminated:
                ipythondisplay.clear_output(wait=True)
                frame = self.env.render()
                plt.imshow(frame)
                plt.axis('off')
                plt.show()
                time.sleep(1.0)

                q_stand = Q_dict.get((state, 0), 0)
                q_hit = Q_dict.get((state, 1), 0)

                if q_stand > q_hit:
                    action = 0
                else:
                    action = 1

                state, reward, terminated, _, _ = self.env.step(action)

            ipythondisplay.clear_output(wait=True)
            frame = self.env.render()
            plt.imshow(frame)
            plt.axis('off')
            plt.show()
            print(f"Episode finished with reward: {reward}")

            if reward > 0:
                wins += 1
            elif reward < 0:
                losses += 1
            else:
                draws += 1

            time.sleep(2)

        self.env.close()
        print(f"\nEvaluation over {TEST_EPISODES} episodes:")
        print(f"  Win Rate: {wins / TEST_EPISODES:.2%}")
        print(f"  Loss Rate: {losses / TEST_EPISODES:.2%}")
        print(f"  Draw Rate: {draws / TEST_EPISODES:.2%}")

    def play_game(self):
        train = self.train_agent()
        evaluate = self.evaluate_agent(train)
        return evaluate

    def td_agent(self, alpha = 0.1, gamma=1, training_episodes=500_000):

        # intialize value function
        V = defaultdict(float)
        # train agent
        for episode in range(training_episodes):
            state, _= self.env.reset()
        terminated = False
        while not terminated:
            # random action based on uniform policy
            action = env.action_space.sample()
            # take action and get next state, reward, terminated, info
            next_state, reward, terminated, _, _ = env.step(action)
            # update value function
            V[state] += alpha * (reward + (gamma * V[next_state]) - V[state])
            # set state to next state
            state = next_state
        return V



