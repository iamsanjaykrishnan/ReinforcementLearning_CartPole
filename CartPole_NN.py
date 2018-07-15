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
        self.tensorflow_summary_init()
        self.DataFrame_init()
        self.i =0
        self.j =0



    def Actor(self,input_tensor):
        init_vs = tf.initializers.variance_scaling(scale=0.01)
        with tf.variable_scope('Actor')as scope:
            hidden = tf.layers.dense(input_tensor, 10, activation=tf.nn.elu,kernel_initializer=init_vs)
            hidden = tf.contrib.layers.layer_norm(hidden)
            hidden = tf.layers.dropout(hidden,self.Dropout)
            hidden = tf.layers.dense(hidden, 10, activation=tf.nn.elu,kernel_initializer=init_vs)
            hidden = tf.contrib.layers.layer_norm(hidden)
            hidden = tf.layers.dropout(hidden, self.Dropout)
            hidden = tf.layers.dense(hidden, 2, activation=tf.nn.elu,kernel_initializer=init_vs)
            hidden = tf.nn.softmax(hidden)
            output = hidden
        return output
    def Critic(self,input_tensor,reuse=0):
        init_vs = tf.initializers.variance_scaling(scale=0.01)
        with tf.variable_scope('Critic')as scope:
            if reuse:
                scope.reuse_variables()
            hidden = tf.layers.dense(input_tensor,10,activation=tf.nn.elu,kernel_initializer=init_vs)
            hidden = tf.contrib.layers.layer_norm(hidden)
            hidden = tf.layers.dropout(hidden,self.Dropout)
            hidden = tf.layers.dense(hidden, 10, activation=tf.nn.elu,kernel_initializer=init_vs)
            hidden = tf.contrib.layers.layer_norm(hidden)
            hidden = tf.layers.dropout(hidden, self.Dropout)
            hidden = tf.layers.dense(hidden, 1, activation=None,kernel_initializer=init_vs)
            output = hidden
        return output

    def define_neural_network(self):
        # reset the graph
        tf.reset_default_graph()

        # define input place holder
        self.State_tensor = tf.placeholder(tf.float32, shape=[None, 4],name='State_tensor' ) # step , state
        self.Action_tensor = tf.placeholder(tf.float32, shape=[None, 2],name='Action_tensor' ) # step , binary encoded action
        self.Reward_tensor = tf.placeholder(tf.float32, shape=[None, 1],name='Reward_tensor' )  # step , binary encoded action
        self.Dropout = tf.placeholder(tf.float32,name='Dropout') # Dropout
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
        CriticLoss_event = tf.losses.mean_squared_error(self.CriticScore_event, self.Reward_tensor)
        self.CriticLossSummary = tf.summary.scalar('Critic_Loss', CriticLoss_event)
        # Actor loss based on generate actions
        ActorLoss = -tf.reduce_mean(self.CriticScore_nn)
        self.ActorLossSummary = tf.summary.scalar('Actor_Loss', ActorLoss)
        # Critic optimisation by reducing prediction error
        self.TrainCritic_op = tf.train.AdamOptimizer().minimize(CriticLoss_event, var_list=C_var)

        # Actor optimisation by maximizing predicted reward
        self.TrainActor_op = tf.train.AdamOptimizer().minimize(ActorLoss, var_list=A_var)

        # init session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def DataFrame_init(self):
        column_names = ['Episode_no','Step_no','Old_State0','Old_State1','Old_State2','Old_State3','New_State0','New_State1','New_State2','New_State3','Action0','Action1','Reward']
        self.df = pd.DataFrame(np.zeros(shape=[1,13],dtype=np.float32),columns=column_names)
        self.logger.info(column_names)

    def DataFrame_append(self,Episode_no,Step_no,OldState,NewState,Action,Reward):
        column_names = ['Episode_no', 'Step_no', 'Old_State0', 'Old_State1', 'Old_State2', 'Old_State3', 'New_State0',
                        'New_State1', 'New_State2', 'New_State3', 'Action0', 'Action1', 'Reward']
        Episode_no = np.reshape(Episode_no,[1,-1])
        Step_no = np.reshape(Step_no,[1,-1])
        OldState = np.reshape(OldState,[1,-1])
        NewState = np.reshape(NewState, [1, -1])
        enc = OneHotEncoder()
        enc.fit([[0], [1]])
        Action = enc.transform(Action).toarray()
        Action = np.reshape(Action,[1,-1])
        Reward = np.reshape(Reward,[1,-1])
        value = np.concatenate([Episode_no,Step_no,OldState,NewState,Action,Reward],1)
        self.logger.info(value)
        appended_values = pd.DataFrame(value, columns=column_names)
        self.df = self.df.append(appended_values,ignore_index=True)

    def save_df(self,csv_file):
        #csv_file = self.nnDirectory+'\\pdFrame.csv'
        self.df.to_csv(csv_file)

    def load_dataframe(self,file):
        self.df = pd.DataFrame.from_csv(file)

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

    def tensorflow_summary_init(self):
        import time as Time
        self.train_writer = tf.summary.FileWriter(self.nnDirectory+ '\\TFsummary\\' + Time.strftime('%H_%M_%S'), self.sess.graph)


    def Critic_init_train(self,iterations):
        OldState = self.df.as_matrix(['Old_State0', 'Old_State1', 'Old_State2', 'Old_State3'])
        Action = self.df.as_matrix(['Action0', 'Action1'])
        Reward = self.df.as_matrix(['Reward'])
        for i in range(iterations):
            feed_dict = {self.State_tensor: OldState, self.Action_tensor: Action,self.Reward_tensor:Reward, self.Dropout: 0.9 }
            _, lossS = self.sess.run([self.TrainCritic_op, self.CriticLossSummary], feed_dict=feed_dict)
            self.i += 1
            self.train_writer.add_summary(lossS, self.i)

    def Actor_train(self,iterations):

        for i in range(iterations):
            dfSample = self.df.sample(25)  # minibatch 25
            OldState = dfSample.as_matrix(['Old_State0', 'Old_State1', 'Old_State2', 'Old_State3'])

            feed_dict = {self.State_tensor: OldState, self.Dropout: 0.9}
            _, lossS = self.sess.run([self.TrainActor_op, self.ActorLossSummary], feed_dict=feed_dict)
            self.j+=1
            self.train_writer.add_summary(lossS,self.j )

    def Critic_value(self,state,action):
        OldState = state
        Action = action
        feed_dict = {self.State_tensor: OldState, self.Action_tensor: Action, self.Dropout: 1}
        Score = self.sess.run([self.CriticScore_event], feed_dict=feed_dict)
        return Score[0]

    def Action(self,State):
        feed_dict = {self.State_tensor: State, self.Dropout: 1}
        Action_one_hot_encoded= self.sess.run([self.nnActionTensor], feed_dict=feed_dict)
        return Action_one_hot_encoded[0]

    def Critic_train(self,iterations):

        for i in range(iterations):
            dfSample = self.df.sample(25) # minibatch 25
            OldState = dfSample.as_matrix(['Old_State0', 'Old_State1', 'Old_State2', 'Old_State3'])
            NewState = dfSample.as_matrix(['New_State0', 'New_State1', 'New_State2', 'New_State3'])
            Action = dfSample.as_matrix(['Action0', 'Action1'])
            Reward = dfSample.as_matrix(['Reward'])
            NextAction = self.Action(NewState)
            NextAction11 = np.ones(np.shape(NextAction))
            NextAction10 = np.copy(NextAction11)
            NextAction10[:, 1] = 0
            NextAction01 = np.copy(NextAction11)
            NextAction01[:, 0] = 0
            PredValue10 = self.Critic_value(NewState, NextAction10)
            PredValue01 = self.Critic_value(NewState, NextAction01)
            PredValues = np.maximum(PredValue10, PredValue01)

            QValue = Reward + np.multiply(Reward, PredValues)  # actual + prediction

            feed_dict = {self.State_tensor: OldState, self.Action_tensor: Action,self.Reward_tensor:QValue, self.Dropout: 0.9  }
            _, lossS = self.sess.run([self.TrainCritic_op, self.CriticLossSummary], feed_dict=feed_dict)
            self.i+=1
            self.train_writer.add_summary(lossS, self.i)

    def SaveNeuralNetwork(self):
        saver = tf.train.Saver()
        # Save the variables in session  to disk.
        saver.save(self.sess, self.nnDirectory+'\\model\CartpoleNN')
    def LoadNeuralNetwork(self,Location):
        # load meta graph and restore weights
        saver = tf.train.import_meta_graph(self.nnDirectory+'\\model\CartpoleNN.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint(self.nnDirectory+'\\model'))


