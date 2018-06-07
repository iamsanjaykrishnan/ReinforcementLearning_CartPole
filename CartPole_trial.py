import numpy as np
from CartPole_NN import NeuralNetwork
import gym
import logging
import os

# clear old log files
os.remove("./Cartpole.log")

# initiate logger
logger = logging.getLogger('Cartpole')
hdlr = logging.FileHandler('./Cartpole.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)

# initialize AI gym for cartpole
env = gym.make('CartPole-v0')
env.reset()

# initialize custom neural network class
NN = NeuralNetwork()

# The rewards and observation of each episode is stored as list to be used as replay in DQN
Reward_trial = []
Observation_trial = []
Action_trial = []

for T in range(20): # Take Random Actions for n trials
    logger.info('Random trials')
    observation_old = env.reset()
    observations_episode = []
    rewards_epidosde = []
    action_epidosde = []
    for t in range(1000):
        #env.render()
        action = env.action_space.sample()
        observation_new, reward, done, info = env.step(action)
        observations_episode.append(observation_old)
        rewards_epidosde.append(reward)
        action_epidosde.append(action)
        observation_old = observation_new
        if done:
            print("Trial {} : Episode finished after {} timesteps".format(T+1,t + 1))
            logger.info("Trial {} : Episode finished after {} timesteps".format(T+1,t + 1))

            # list -> np
            rewards_epidosde = np.asarray(rewards_epidosde)
            observations_episode = np.asarray(observations_episode)
            rewards_epidosde = rewards_epidosde.reshape(-1,1)
            action_epidosde = np.asarray(action_epidosde)
            action_epidosde = action_epidosde.reshape(-1, 1)
            logger.info("rewards_epidosde.shape {}" .format(rewards_epidosde.shape))
            logger.info("action_epidosde.shape {}" .format(action_epidosde.shape))

            # Calculate cumsum for reward in desending order
            rewards_epidosde = np.flipud(rewards_epidosde)
            rewards_epidosde = np.cumsum(rewards_epidosde)
            rewards_epidosde = np.flipud(rewards_epidosde)
            logger.info("rewards_epidosde {}".format(rewards_epidosde))
            logger.info("action_epidosde {}".format(action_epidosde))
            break

    # append observation and reward of episode to list for later use
    Observation_trial.append(observations_episode)
    Reward_trial.append(rewards_epidosde)

# Teach neural network to play
for Trial in range(len(Observation_trial)): # iterate for all previous trials
    time_steps, _ = Observation_trial[Trial].shape
    NN.trainDiscriminator(Observation_trial[Trial],Action_trial[Trial],Reward_trial[Trial] )
env.close()