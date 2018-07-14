from CartPole_NN import NeuralNetwork
import gym

# initialize AI gym for cartpole
env = gym.make('CartPole-v0')
env.reset()
# initialize custom neural network class
NN = NeuralNetwork()

for E in range(100): # Take Random Actions for n trials and collect data
    observation_old = env.reset()
    for s in range(1000):
        #env.render()
        action = env.action_space.sample()
        observation_new, reward, done, info = env.step(action)
        if done:
            reward = 0
        else:
            reward = 1
        NN.DataFrame_append(Episode_no=E+1,Step_no=s,OldState=observation_old,NewState=observation_new,Action=action,Reward=reward)
        observation_old = observation_new
        if done:
            print("Trial {} : Episode finished after {} timesteps".format(E+1,s + 1))
            break

NN.save_df() # Save data frame as csv

env.close()