# RL-Gymnasium-Agents
**RL Methods implemented from scratch on Gymnasium Libraries.**

In this project, I built an agent that plays Blackjack based on the Monte Carlo First Visit Method with Exploring Starts.

The project starts by creating the environment and visualizing the sample action and observation spaces. We set the render mode to rgb_array so we can visualize the agent playing the game.

Through initializing the Q_dict, we are able to record state-action pairs as keys with their values being the calculated returns, updated iteratively once an episode terminates. The purpose of the freq dictionary is to record state-action pairs and the frequency at which first visits occur so we can calculate the average return, correctly implementing the Monte Carlo method.
