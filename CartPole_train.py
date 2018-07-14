from CartPole_NN import NeuralNetwork
import gym
import numpy as np


# initialize AI gym for cartpole
env = gym.make('CartPole-v0')
env.reset()
# preCollected data
data = 'E:\ReinforcementLearning_CartPole\\NN_data\\pdFrame.csv'
# initialize custom neural network class
NN = NeuralNetwork()
# load old data
NN.load_dataframe(data)
# Train critic initial
NN.Critic_init_train(10000)
NN.Critic_train(1000)
NN.Actor_train(1000)

for E in range(20):
    observation_old = env.reset()
    for s in range(200):
        env.render()
        observation_old = np.reshape(observation_old,[1,-1])
        action_one_hot = NN.Action(observation_old)
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
            NN.Critic_train(1000)
            NN.Actor_train(1000)
            break

#NN.save_df(NN.nnDirectory+'\\120Episodes.csv')
NN.SaveNeuralNetwork()