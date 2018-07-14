import tensorflow as tf
import numpy as np
import logging
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os

class NeuralNetwork:
    def __init__(self):
        self.path_init()
        self.init_logger()
        self.define_neural_network()
        self.DataFrame_init()


    def Actor(self,input_tensor):
        init_vs = tf.initializers.variance_scaling(scale=0.05)
        with tf.variable_scope('Actor')as scope:
            hidden = tf.layers.dense(input_tensor, 10, activation=tf.nn.tanh,kernel_initializer=init_vs)
            hidden = tf.layers.dense(hidden, 10, activation=tf.nn.tanh,kernel_initializer=init_vs)
            hidden = tf.layers.dense(hidden, 2, activation=tf.nn.sigmoid,kernel_initializer=init_vs)
            output = hidden
        return output
    def Critic(self,input_tensor,reuse=0):
        init_vs = tf.initializers.variance_scaling(scale=0.05)
        with tf.variable_scope('Critic')as scope:
            if reuse:
                scope.reuse_variables()
            hidden = tf.layers.dense(input_tensor,10,activation=tf.nn.tanh,kernel_initializer=init_vs)
            hidden = tf.layers.dense(hidden, 10, activation=tf.nn.tanh,kernel_initializer=init_vs)
            hidden = tf.layers.dense(hidden, 1, activation=None,kernel_initializer=init_vs)
            output = hidden
        return output

    def define_neural_network(self):
        # reset the graph
        tf.reset_default_graph()

        # define input place holder
        self.State_tensor = tf.placeholder(tf.float32, shape=[None, 4]) # step , state
        self.Action_tensor = tf.placeholder(tf.float32, shape=[None, 2]) # step , binary encoded action
        self.Reward_tensor = tf.placeholder(tf.float32, shape=[None, 1])  # step , binary encoded action

        # define Actor
        self.nnActionTensor = self.Actor(self.State_tensor)
        print()

        # define Critic based on event Action tensor
        self.CriticStateAction_event = tf.concat([self.State_tensor,self.Action_tensor],1)
        self.CriticScore_event = self.Critic(self.CriticStateAction_event)

        # define critic based on nnActionTensor
        self.CriticState_nnAction = tf.concat([self.State_tensor, self.nnActionTensor],1)
        self.CriticScore_nn = self.Critic(self.CriticState_nnAction,1) # reuse the previously defined weights

        # Define Training methods
        # Get variables to be trained
        C_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Critic')
        A_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Actor')

        # Critic loss based on event data
        print(self.CriticScore_event)
        print(self.Reward_tensor)
        CriticLoss_event = tf.losses.mean_squared_error(self.CriticScore_event, self.Reward_tensor)
        # Actor loss based on generate actions
        ActorLoss = -self.CriticScore_nn

        # Critic optimisation by reducing prediction error
        self.TrainCritic_op = tf.train.AdamOptimizer().minimize(CriticLoss_event, var_list=C_var)

        # Actor optimisation by maximizing predicted reward
        self.TrainActor_op = tf.train.AdamOptimizer().minimize(ActorLoss, var_list=A_var)

        # init session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def DataFrame_init(self):
        column_names = ['Episode_no','Step_no','State0','State1','State2','State3','Action0','Action1','Reward']
        self.df = pd.DataFrame(np.zeros(shape=[1,9],dtype=np.float32),columns=column_names)
        self.logger.info(column_names)

    def DataFrame_append(self,Episode_no,Step_no,State,Action,Reward):
        column_names = ['Episode_no', 'Step_no', 'State0', 'State1', 'State2', 'State3', 'Action0', 'Action1', 'Reward']
        Episode_no = np.reshape(Episode_no,[1,-1])
        Step_no = np.reshape(Step_no,[1,-1])
        State = np.reshape(State,[1,-1])
        enc = OneHotEncoder()
        enc.fit([[0], [1]])
        Action = enc.transform(Action).toarray()
        Action = np.reshape(Action,[1,-1])
        Reward = np.reshape(Reward,[1,-1])
        value = np.concatenate([Episode_no,Step_no,State,Action,Reward],1)
        self.logger.info(value)
        appended_values = pd.DataFrame(value, columns=column_names)
        self.df = self.df.append(appended_values,ignore_index=True)

    def save_df(self):
        csv_file = self.nnDirectory+'\\pdFrame.csv'
        self.df.to_csv(csv_file)

    def path_init(self):
        self.cwd = os.getcwd()
        directory = str(self.cwd)+('\\NN_data')
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.nnDirectory = directory

    def init_logger(self):
        # clear old log files
        log_file =self.nnDirectory + '\\CartpoleNN.log'
        try:
            os.remove(log_file)
        except:
            pass
        # initiate logger
        logger = logging.getLogger('CartpoleNN')
        hdlr = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        logger.setLevel(logging.DEBUG)
        self.logger = logger