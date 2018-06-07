import tensorflow as tf
import numpy as np
import logging
import os

# clear old log files
os.remove("./CartpoleNN.log")

# initiate logger
logger = logging.getLogger('CartpoleNN')
hdlr = logging.FileHandler('./CartpoleNN.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)

class NeuralNetwork:
    def __init__(self): # constructor
        self.neural_network_init()
    def Actor(self,input_actor):
        # define Actor neural network
        with tf.variable_scope('Generator')as scope:
            hidden = tf.layers.dense(input_actor, units=10, )
            hidden = tf.sigmoid(hidden) * hidden
            hidden = tf.layers.dense(hidden, units=10, )
            hidden = tf.sigmoid(hidden) * hidden
            hidden = tf.layers.dense(hidden, units=10, )
            hidden = tf.sigmoid(hidden) * hidden
            hidden = tf.layers.dense(hidden, units=2, )
            Actor_out = hidden
        return Actor_out


    def Discrimitator(self,input_disc,reuse=False):
        # define Discrimitator neural network
        with tf.variable_scope('Discriminator')as scope:
            if reuse:
                scope.reuse_variables()
            hidden = tf.layers.dense(input_disc, units=10, )
            hidden = tf.sigmoid(hidden) * hidden
            hidden = tf.layers.dense(hidden, units=10, )
            hidden = tf.sigmoid(hidden) * hidden
            hidden = tf.layers.dense(hidden, units=10, )
            hidden = tf.sigmoid(hidden) * hidden
            hidden = tf.layers.dense(hidden, units=1, )
            Reward_prediction = hidden
        return Reward_prediction

    def neural_network_init(self):
        # reset the graph
        tf.reset_default_graph()

        # define input place holder
        self.Input_placeholder = tf.placeholder(tf.float32, shape=[None, 4 , 10])
        self.Observation = tf.layers.flatten(self.Input_placeholder)
        print( self.Observation)
        self.Action_past = tf.placeholder(tf.float32, shape=[None, 2])
        self.Reward = tf.placeholder(tf.float32, shape=[None, 1])

        # build Actor->Discriminator connected neural network
        actor_descision = self.Actor(self.Observation)
        Act_disc_inp = tf.concat([self.Observation, actor_descision],axis=1)
        Act_reward_pred = self.Discrimitator(Act_disc_inp)

        # Create discriminator that reuses the weights
        # This network can be trained independantly
        disc_inp = tf.concat([self.Observation, self.Action_past],axis=1)
        disc_out = self.Discrimitator(disc_inp, True)

        # Define Training methods
        # Get variables to be trained
        D_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Discriminator')
        G_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Generator')
        # Discriminator loss
        Reward_loss = tf.losses.mean_squared_error(self.Reward, disc_out)
        # Discriminator optimisation by reducing prediction error
        self.Disc_op = tf.train.AdamOptimizer().minimize(Reward_loss, var_list=D_var)
        # Actor optimisation by maximizing predicted reward
        self.Actor_op = tf.train.AdamOptimizer().minimize(-Act_reward_pred, var_list=G_var)

    def Input_One_hot_encoded(self, state, axis = 2):
        # Create one hot encoded bins for State
        def Binning(x):
            no_bins = 10
            bin_boundry = np.linspace(-1,1,no_bins)
            out_,_ = np.histogram(x, bin_boundry)
            return out_
        return np.apply_along_axis(Binning, axis, state)

    def trainDiscriminator(self,Observation_trial,Action_trial,Reward_trial,iteration =100):
        Observation_trial = np.asarray(Observation_trial)
        #Observation_trial =[trials,timesteps,state] axis 2 to be encoded
        Observation_trial = self.Input_One_hot_encoded(Observation_trial,2)
        Action_trial = np.asarray(Action_trial)
        Reward_trial = np.asarray(Reward_trial)
        for _ in range(iteration):
            pass
        pass
