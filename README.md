
# **RL Methods implemented from scratch on Gymnasium Libraries.**

In this project we have Monte Carlo methods, TD On-Policy and Off-Policy Methods, as well as Value-Based DRL Methods.

I built an agent that plays Blackjack based on the Monte Carlo First Visit Method with Exploring Starts. Implemented TD_learning with SARSA for taxi, and built a DQN for Cartpole.

# MC Blackjack
The project starts by creating the environment and visualizing the sample action and observation spaces. We set the render mode to rgb_array so we can visualize the agent playing the game.

Through initializing the Q_dict, we are able to record state-action pairs as keys. The keys have values being the calculated returns, which are updated iteratively after termination of the episode. Before we can do that though, we need a list that records each visit to the next state, the action taken and reward received as a tuple for each episode. 

The magic really happens when our list is reversed, we take the last state tuple and dissect it. Every state moving backwards from the last will be updated with NewEstimate = OldEstimate + (Target - OldEstimate) / Count where Target is G and the OldEstimate is simply the current value of the state-action pair in the Q_dict before the update is performed. Each state action pair is only updated when it is visited in the current episode. This is why we have the freq dictionary: to record state-action pairs and the frequency at which first visits occur in each episode so we can calculate the average return, correctly implementing the Monte Carlo method.

The algorithm can be found in Sutton and Barto Reinforcement Learning 2nd edition Ch 5.3 Page 99: 

<img width="1049" height="548" alt="Screenshot 2025-09-15 at 11 32 20 AM" src="https://github.com/user-attachments/assets/d8c2d299-2f1b-4a44-a0e6-b80a00a24dfa" />

# Results

After training the agent on 50,000 episodes we now test this on ten games which are rendered for our enjoyment. The results are as shown below: 


<img width="475" height="505" alt="Screenshot 2025-09-15 at 4 54 56 PM" src="https://github.com/user-attachments/assets/4995b975-3282-4f69-9eaa-3d7832ece8f1" />


# **Deep Q-Network (DQN) — CartPole-v1**

 The network structure consists of three fully connected layers with ReLU activations. He (Kaiming) weight initialization is used to stabilize gradients, since we use Relu activation functions. Epsilon-greedy exploration decays exponentially by 0.995 each episode and capped at a lower bound of 0.17. The cap is based on experimental results showing a sharp performance decline below this threshold.

The optimizer is RMSProp, chosen over Adam due to its ability to handle non-stationary and high-variance data, as Adam’s momentum term sometimes destabilized training in previous experiments. The agent uses a uniform replay buffer of up to 10,000 transitions, with minibatches of 64 drawn after a warmup period of 1,000 samples. A target network is maintained and synchronized with the policy network every 1,000 steps. The loss function is mean squared error between current and target Q-values.

Limitations include the lack of prioritized experience replay, which would focus learning on more important transitions, and the absence of Double DQN (DDQN), which reduces overestimation bias in Q-learning. Incorporating these techniques would likely improve stability and sample efficiency.

During training, average rewards increased from about r = 11 at the start to r > 220 by episode 300, peaking over r = 300 during episodes 350-400 on average. After epsilon reached its minimum value, average rewards hovered between 170 and 230. The results confirm the agent is learning, but also indicate further optimization is required.

The pseudocode implementation is written as follows:

<img width="972" height="497" alt="Screenshot 2025-10-11 at 5 56 01 PM" src="https://github.com/user-attachments/assets/6a7c9c41-0253-4de9-b34b-0870e582b195" />




