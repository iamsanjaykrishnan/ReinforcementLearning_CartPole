import numpy as np
from CartPole_NN import NeuralNetwork
import gym
import logging
import os

# initialize AI gym for cartpole
env = gym.make('CartPole-v0')
env.reset()

# initialize custom neural network class
NN = NeuralNetwork()

# The rewards and observation of each episode is stored as list to be used as replay in DQN
Reward_trial = []
Observation_trial = []
Action_trial = []

for T in range(20): # Take Random Actions for n trials and collect data
    observation_old = env.reset()
    for t in range(1000):
        #env.render()
        action = env.action_space.sample()
        observation_new, reward, done, info = env.step(action)
        if done:
            reward = 0
        NN.DataFrame_append(Episode_no=T,Step_no=t,State=observation_new,Action=action,Reward=reward)
        if done:
            print("Trial {} : Episode finished after {} timesteps".format(T+1,t + 1))
            break

NN.save_df()

env.close()