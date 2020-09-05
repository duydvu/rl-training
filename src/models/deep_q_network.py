# -*- coding: utf-8 -*-
from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Dense, Activation, Input, Conv2D, Flatten, MaxPool2D, multiply
from keras import optimizers
from keras import backend as K
import tensorflow as tf
from random import random, randrange


# Deep Q Network off-policy
class DQN: 
    def __init__(
            self,
            input_max_x, #The number of inputs for the DQN network
            input_max_y, #The number of inputs for the DQN network
            action_space, #The number of actions for the DQN network
            gamma = 0.99, #The discount factor
            epsilon = 1, #Epsilon - the exploration factor
            epsilon_min = 0.01, #The minimum epsilon 
            epsilon_decay = 0.999,#The decay epislon for each update_epsilon time
            learning_rate = 0.00025, #The learning rate for the DQN network
            tau = 0.125, #The factor for updating the DQN target network from the DQN network
            model = None, #The DQN model
            target_model = None, #The DQN target model 
            sess = None,
            from_checkpoint=None,
    ):
        self.input_max_x = input_max_x
        self.input_max_y = input_max_y
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau
        self.from_checkpoint = from_checkpoint
              
        #Creating networks
        self.model = self.create_model() #Creating the DQN model
        self.target_model = self.create_model() #Creating the DQN target model
        self.model.summary()
        
        #Tensorflow GPU optimization
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        K.set_session(tf.Session(config=config))

    def create_model(self):
        input_view = Input(shape=(self.input_max_x, self.input_max_y, 5))
        conv = Conv2D(filters=32, kernel_size=4,
                      strides=1, activation='relu')(input_view)
        # max_pool = MaxPool2D(pool_size=2, strides=2)(conv)
        conv2 = Conv2D(filters=64, kernel_size=2,
                       strides=2, activation='relu')(conv)
        conv3 = Conv2D(filters=128, kernel_size=2,
                       strides=2, activation='relu')(conv2)
        max_pool3 = MaxPool2D(pool_size=[3, 2], strides=[2, 1])(conv3)
        flatten = Flatten()(max_pool3)
        y = Dense(128, activation='relu')(flatten)
        model1 = Model(inputs=input_view, outputs=y)

        input_energy = Input(shape=(1,))
        model2_output = Dense(128, activation='tanh')(input_energy)
        model2 = Model(inputs=input_energy, outputs=model2_output)

        mul = multiply([model1.output, model2.output])

        output = Dense(self.action_space, activation='linear')(mul)

        model = Model(inputs=[input_view, input_energy], outputs=output)
        adam = optimizers.adam(lr=self.learning_rate)
        model.compile(optimizer=adam, loss='mse')
        if self.from_checkpoint:
            model.load_weights('%s.h5' % self.from_checkpoint[:-5])
        return model


    @staticmethod
    def transform_state(state):
        view, players = state
        model_view = np.zeros((view.shape[0] + 2, view.shape[1] + 2, 5), dtype=float)
        model_view[:, :, 0] = -5
        model_view[1:-1, 1:-1, 0] = view
        model_view[model_view > 0] = model_view[model_view > 0] / 1000
        for player in players:
            if player['id'] == 1:
                energy = player['energy'] / 50
            x = player['x']
            y = player['y']
            model_view[x + 1, y + 1, player['id']] = 1
        return [model_view, [energy]]
  
    
    def act(self, state):
        model_state = self.transform_state(state)
        a_max = np.argmax(self.model.predict([[model_state[0]], [model_state[1]]]))      
        if (random() < self.epsilon):
            a_chosen = randrange(self.action_space)
        else:
            a_chosen = a_max      
        return a_chosen
    
    
    def replay(self, samples):
        states, actions, rewards, new_states, dones = samples
        states = list(zip(*map(self.transform_state, states)))
        states = [list(states[0]), list(states[1])]
        new_states = list(zip(*map(self.transform_state, new_states)))
        new_states = [list(new_states[0]), list(new_states[1])]
        targets = self.target_model.predict(states)
        Q_futures = np.max(self.target_model.predict(new_states),
                           axis=1)

        for i in range(len(states)):
            action = actions[i]
            reward = rewards[i]
            done = dones[i]
            if done:
                targets[i,action] = reward # if terminated, only equals reward
            else:
                Q_future = Q_futures[i]
                targets[i,action] = reward + Q_future * self.gamma

        #Training
        loss = self.model.train_on_batch(states, targets)
    
    def target_train(self): 
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(0, len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        
        self.target_model.set_weights(target_weights) 
    
    
    def update_epsilon(self):
        self.epsilon =  self.epsilon*self.epsilon_decay
        self.epsilon =  max(self.epsilon_min, self.epsilon)
    
    
    def save_model(self, model_path):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(model_path + ".json", "w") as json_file:
            json_file.write(model_json)
            # serialize weights to HDF5
        self.model.save_weights(model_path + ".h5")
        print("Saved model to disk")
