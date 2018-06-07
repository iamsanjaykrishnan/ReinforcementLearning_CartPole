import numpy as np
from CartPole_NN import NeuralNetwork
import gym
import logging

logger = logging.getLogger('Cartpole')
hdlr = logging.FileHandler('./Cartpole.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)

env = gym.make('CartPole-v0')
env.reset()

NN = NeuralNetwork()

Reward_trial = []
Observation_trial = []
# Take Random Actions for 10 trials
for T in range(20):
    logger.info('Random trials')
    observation_old = env.reset()
    observations_episode = []
    rewards_epidosde = []
    for t in range(1000):
        env.render()
        action = env.action_space.sample()
        observation_new, reward, done, info = env.step(action)
        observations_episode.append(observation_old)
        rewards_epidosde.append(reward)
        observation_old = observation_new
        if done:
            print("Trial {} : Episode finished after {} timesteps".format(T+1,t + 1))
            logger.info("Trial {} : Episode finished after {} timesteps".format(T+1,t + 1))
            # Calculate list to np
            rewards_epidosde = np.asarray(rewards_epidosde)
            observations_episode = np.asarray(observations_episode)
            rewards_epidosde = rewards_epidosde.reshape(-1,1)
            logger.info("rewards_epidosde.shape"+str(rewards_epidosde.shape))
            # Calculate cumsum from the end
            rewards_epidosde = np.flipud(rewards_epidosde)
            rewards_epidosde = np.cumsum(rewards_epidosde)
            rewards_epidosde = np.flipud(rewards_epidosde)
            logger.info("rewards_epidosde"+str(rewards_epidosde))
            break
    Observation_trial.append(observations_episode)
    Reward_trial.append(rewards_epidosde)
env.close()