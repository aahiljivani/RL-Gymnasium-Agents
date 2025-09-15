# RL-Gymnasium-Agents
**RL Methods implemented from scratch on Gymnasium Libraries.**

In this project, I built an agent that plays Blackjack based on the Monte Carlo First Visit Method with Exploring Starts.

The project starts by creating the environment and visualizing the sample action and observation spaces. We set the render mode to rgb_array so we can visualize the agent playing the game.

Through initializing the Q_dict, we are able to record state-action pairs as keys. The keys have values being the calculated returns, which are updated iteratively after termination of the episode. Before we can do that though, we need a list that records each visit to the next state, the action taken and reward received as a tuple for each episode. 

The magic really happens when our list is reversed, we take the last state tuple and dissect it. Every state moving backwards from the last will be updated with NewEstimate = OldEstimate + (Target - OldEstimate) / Count where Target is G and the OldEstimate is simply the current value of the state-action pair in the Q_dict before the update is performed. Each state action pair is only updated when it is visited in the current episode. This is why we have the freq dictionary: to record state-action pairs and the frequency at which first visits occur in each episode so we can calculate the average return, correctly implementing the Monte Carlo method.

The algorithm can be found in Sutton and Barto Reinforcement Learning 2nd edition Ch 5.3 Page 99: 

<img width="1049" height="548" alt="Screenshot 2025-09-15 at 11 32 20 AM" src="https://github.com/user-attachments/assets/d8c2d299-2f1b-4a44-a0e6-b80a00a24dfa" />

# Results

After training the agent on 50,000 episodes we now test this on ten games which are rendered for our enjoyment. The results are as shown below: 


<img width="475" height="505" alt="Screenshot 2025-09-15 at 4 54 56 PM" src="https://github.com/user-attachments/assets/4995b975-3282-4f69-9eaa-3d7832ece8f1" />


