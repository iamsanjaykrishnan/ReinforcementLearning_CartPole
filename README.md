# CartPole Reinforcement Learning 

![ReinforcementLearning_Sanjay Krishnan Venugopal](https://github.com/iamsanjaykrishnan/ReinforcementLearning_CartPole/blob/master/SanjayReinforcementLearning.gif)<br />
Exploration : <br />
- Based on random action for 100 episodes.<br />
Training : <br />
- Training is done on explored data.<br />
- Reward function is defined as the number of time steps survived.<br />
- It is trained based on actor critic algorithm, a GAN based algorithm. <br />
- The critic network models the Q value and the actor network tries to maximize the reward for all possible states that has been explored <br />
Qvalue = reward_for_current_step + Max_Qvalue_for_next_step<br /><br />
Hyper parameters<br />
Activation function : Elu
Regularization : Droupout<br />
Optimizer : Adam<br />
Network : Actor(State>10>10>2-Action), Critic(State+Action>10>10>1-Qvalue)<br />
Normalization : layer normalization<br />
<br />
Result during training: (training 20 episodes based on explored data)<br />
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
<br /><br />
The critic network was initallized by trained with explored data. Unexplored points were interpolated based on elu function. The explored data was limited to a max of 40 steps and to improve the critic network, the 20 episodes resulting during training of the network and explored data are used to reinitilize Q value predicted from critic network. This results in better training.
<br />
# Architecture -> Actor Critic
![ReinforcementLearning_A2C](https://github.com/iamsanjaykrishnan/ReinforcementLearning_CartPole/blob/master/NetworkArchitecture.png)
