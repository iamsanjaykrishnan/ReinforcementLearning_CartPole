from CartPole_NN import NeuralNetwork
import gym

# initialize AI gym for cartpole
env = gym.make('CartPole-v0')
env.reset()
# initialize custom neural network class
NN = NeuralNetwork()

for E in range(20): # Take Random Actions for n trials and collect data
    observation_old = env.reset()
    for s in range(1000):
        #env.render()
        action = env.action_space.sample()
        observation_new, reward, done, info = env.step(action)
        if done:
            reward = 0
        NN.DataFrame_append(Episode_no=E+1,Step_no=s,State=observation_old,Action=action,Reward=reward)
        observation_old = observation_new
        if done:
            print("Trial {} : Episode finished after {} timesteps".format(E+1,s + 1))
            break

NN.save_df() # Save data frame as csv

env.close()