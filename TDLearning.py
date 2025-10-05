import gymnasium as gym
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import imageio
from matplotlib import animation
from IPython.display import HTML
from IPython.display import Image


class TD_Control:
    def __init__(self):
        # define the environment, q_pi, action_space length
        self.env = gym.make("Taxi-v3", render_mode='rgb_array')
        self.q_values = defaultdict(lambda: [0.0] * self.env.action_space.n)
        self.action_space = self.env.action_space.n


    def e_greedy_policy(self, state, info, epsilon):

      ''' This is the epsilon-greedy policy which we can use for both SARSA and Q-Learning'''

      # action mask contains the actions that are available in state
      actions = info['action_mask']
      # valid actions carry the index of actions that can be used
      valid_actions = [i for i, v in enumerate(actions) if v == 1]
      # get all action values from state
      action_values = self.q_values[state]
      # implement random action when number is <= epsilon
      if np.random.random() <= epsilon:
          action = np.random.choice(valid_actions)
      else:
        # retrieve all the action values for the valid actions in q_val
          q_val = [action_values[i] for i in valid_actions]
          # argmax returns the index that has the largest action value
          action = valid_actions[np.argmax(q_val)]

      return action

    def sarsa(self, episodes, epsilon=0.1, alpha=0.1, gamma=0.99):

      # Will hold frames only for final episode
      frames = []

      for ep in range(episodes):
        # define state and info only
          state, info = self.env.reset()
          # choose greedy action
          action = self.e_greedy_policy(state, info, epsilon)
          # define terminated and truncated
          terminated = False
          truncated = False

          # Only capture frames for last episode
          capture_frames = (ep == episodes - 1)

          # while episode is still running..
          while not (terminated or truncated):

              if capture_frames:
                  frame = self.env.render()
                  frames.append(frame)

              # contain the next states rewards and termination, info for use of q_pi update
              next_state, reward, terminated, truncated, next_info = self.env.step(action)
              # e_greedy action
              next_action = self.e_greedy_policy(next_state, next_info, epsilon)

              # define state action values of current and next state using results of action
              q_sa = self.q_values[state][action]
              q_next_sa = self.q_values[next_state][next_action]
              # Update q state-action values
              self.q_values[state][action] += alpha * (reward + gamma * q_next_sa - q_sa)

              # update state and action in while loop
              state = next_state
              action = next_action

      # Save only after final episode
      imageio.mimsave('taxi_last_episode.gif', frames, duration=0.2)