from CartPole_NN import NeuralNetwork
import gym
import numpy as np


# initialize AI gym for cartpole
env = gym.make('CartPole-v0')
env.reset()
# preCollected data
data = 'E:\ReinforcementLearning_CartPole\\NN_data\\120Episodes.csv'
# initialize custom neural network class
NN = NeuralNetwork()

# Load trained network
NN.LoadNeuralNetwork()


# Add disturbance
disturb = True
disturbStd = 0.5

for E in range(10):
    observation_old = env.reset()
    for s in range(200):
        env.render()
        observation_old = np.reshape(observation_old,[1,-1])
        action_one_hot = NN.Action(observation_old)
        if disturb == 1:
            action_one_hot += np.random.normal(0,disturbStd,np.shape(action_one_hot))

        if action_one_hot[0][0]>action_one_hot[0][1]:
            action = 0
        else:
            action = 1


        observation_new, reward, done, info = env.step(action)
        if done:
            reward = 0
        else:
            reward = 1
        NN.DataFrame_append(Episode_no=E,Step_no=s,OldState=observation_old,NewState=observation_new,Action=action,Reward=reward)

        observation_old = observation_new
        if done:
            print("Trial {} : Episode finished after {} timesteps".format(E+1,s + 1))
            break
