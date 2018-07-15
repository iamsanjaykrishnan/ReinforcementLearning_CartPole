# ReinforcementLearning_CartPole

![ReinforcementLearning_Sanjay Krishnan Venugopal](https://github.com/iamsanjaykrishnan/ReinforcementLearning_CartPole/blob/master/SanjayReinforcementLearning.gif)<br />
Training method : <br />100 episodes of random action were performed initially. The number of time steps survived is the reward with 200 being the maximum. Under random sampling a max score of approx 40 was achieved. Based on the random samples, the critic network was modelled to predict Q value. The actor network is trained to maximize the predicted Q value.<br />
Q = reward_for_current_step + Max_reward_for_next_step<br /><br />
From the results it is clear that the neural network models a method to maximize score.<br />
The model runs A2C learning. <br />
Result <br />
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
