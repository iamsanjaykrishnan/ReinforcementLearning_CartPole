# CartPole Reinforcement Learning 

![ReinforcementLearning_Sanjay Krishnan Venugopal](https://github.com/iamsanjaykrishnan/ReinforcementLearning_CartPole/blob/master/SanjayReinforcementLearning.gif)<br />
Exploration : Based on random action for 100 episodes.<br />
Training : <br />
- Training is done on explored data.<br />
- Reward function is defined as the number of time steps survived.<br />
	Q = reward_for_current_step + Max_Qvalue_for_next_step<br /><br />
- It is trained based on actor critic algorithm. 
- The critic network models the Q value and the actor network tries to maximize the reward for all possible states that has been explored <br />
Activation function used : Elu<br />
Regularization : Droupout<br />
Optimizer : Adam<br />
Network : Actor(State>10>10>2-Action), Critic(State+Action>10>10>1-Qvalue)<br />
Normalization : layer normalization<br />
<br />
Result during training 20 episodes based on explored data<br />
Trial 1 : Episode finished after 131 timesteps<br />
Trial 2 : Episode finished after 200 timesteps<br />
Trial 3 : Episode finished after 104 timesteps<br />
Trial 4 : Episode finished after 145 timesteps<br />
Trial 5 : Episode finished after 200 timesteps<br />
Trial 6 : Episode finished after 172 timesteps<br />
Trial 7 : Episode finished after 200 timesteps<br />
Trial 8 : Episode finished after 195 timesteps<br />
Trial 9 : Episode finished after 200 timesteps<br />
Trial 10 : Episode finished after 200 timesteps<br />
Trial 11 : Episode finished after 200 timesteps<br />
Trial 12 : Episode finished after 200 timesteps<br />
Trial 13 : Episode finished after 200 timesteps<br />
Trial 14 : Episode finished after 200 timesteps<br />
Trial 15 : Episode finished after 200 timesteps<br />
Trial 16 : Episode finished after 200 timesteps<br />
Trial 17 : Episode finished after 185 timesteps<br />
Trial 18 : Episode finished after 190 timesteps<br />
Trial 19 : Episode finished after 191 timesteps<br />
Trial 20 : Episode finished after 198 timesteps<br />

# ReinforcementLearning_CartPole_Architecture -> Actor Critic or GAN
![ReinforcementLearning_A2C](https://github.com/iamsanjaykrishnan/ReinforcementLearning_CartPole/blob/master/NetworkArchitecture.png)
