# RL-Gymnasium-Agents
**RL Methods implemented from scratch on Gymnasium Libraries.**

In this project, I built an agent that plays Blackjack based on the Monte Carlo First Visit Method with Exploring Starts.

The project starts by creating the environment and visualizing the sample action and observation spaces. We set the render mode to rgb_array so we can visualize the agent playing the game.

Through initializing the Q_dict, we are able to record state-action pairs as keys with their values being the calculated returns, updated iteratively once an episode terminates. The purpose of the freq dictionary is to record state-action pairs and the frequency at which first visits occur so we can calculate the average return, correctly implementing the Monte Carlo method.

The algorithm can be found in Sutton and Barto Reinforcement Learning 2nd edition Ch 5.3 Page 99: 

<img width="1049" height="548" alt="Screenshot 2025-09-15 at 11 32 20 AM" src="https://github.com/user-attachments/assets/d8c2d299-2f1b-4a44-a0e6-b80a00a24dfa" />
