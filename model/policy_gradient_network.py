# -*- coding: utf-8 -*-
from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf
from random import random, randrange


# Deep Q Network off-policy
class PGNetwork: 
    def __init__(
            self,
            view_state_size,
            player_state_size,
            action_size,
            filters,
            lstm_state_size,
            hidden_layer_size,
            gamma = 0.99, #The discount factor
            learning_rate = 0.00025,
            sess = None,
            from_checkpoint=None,
    ):
        self.view_state_size = view_state_size
        self.player_state_size = player_state_size
        self.action_size = action_size
        self.filters = filters
        self.lstm_state_size = lstm_state_size
        self.hidden_layer_size = hidden_layer_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.from_checkpoint = from_checkpoint

        # Create model
        self.view_inputs = tf.placeholder(tf.float32, [None, None, *self.view_state_size], name='view_inputs')
        self.player_inputs = tf.placeholder(tf.float32, [None, None, self.player_state_size], name='player_inputs')
        self.actions = tf.placeholder(tf.int32, [None, None, self.action_size], name="actions")
        self.discounted_episode_rewards = tf.placeholder(
            tf.float32, [None, None], name='discounted_episode_rewards')
        self.length = tf.placeholder(tf.float32, [None,], name='length')
        self.state_placeholder = tf.placeholder(tf.float32, [2, None, self.lstm_state_size])
        
        self.flat_view_inputs = tf.reshape(self.view_inputs, [-1, *self.view_state_size])

        with tf.variable_scope('view'):
            self.conv_out_width = self.view_state_size[0]
            self.conv_out_height = self.view_state_size[1]
            self.conv_layers = []
            for idx, (num_filters, kernel_size, strides) in enumerate(self.filters):
                self.conv_out_width = (self.conv_out_width - kernel_size) / strides + 1
                self.conv_out_height = (self.conv_out_height - kernel_size) / strides + 1
                self.conv_layers.append(self.conv_layer(self.flat_view_inputs,
                                                        filters=num_filters,
                                                        kernel_size=kernel_size,
                                                        strides=strides,
                                                        name='conv-%d' % idx))
            self.conv_out_size = int(self.conv_out_width * self.conv_out_height * filters[-1][0])
            self.flatten = tf.layers.flatten(self.conv_layers[-1], name='flatten')
            self.flatten = tf.reshape(self.flatten, [-1, tf.shape(self.view_inputs)[1], self.conv_out_size])

        with tf.variable_scope('player'):
            self.player_out = tf.layers.dense(inputs=self.player_inputs,
                                              units=self.conv_out_size,
                                              activation=tf.nn.tanh,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              name='layer')

        self.multiply = tf.multiply(self.flatten, self.player_out)

        self.unstack_state = tf.unstack(self.state_placeholder)
        rnn_tuple_state = tf.nn.rnn_cell.LSTMStateTuple(self.unstack_state[0], self.unstack_state[1])
        self.lstm_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_state_size, state_is_tuple=True)
        self.lstm_outs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm_cell,
                                                            self.multiply,
                                                            sequence_length=self.length,
                                                            initial_state=rnn_tuple_state)

        self.fc = tf.layers.dense(inputs=self.lstm_outs,
                                  units=self.hidden_layer_size,
                                  activation=tf.nn.elu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  name='fc1')

        self.logits = tf.layers.dense(inputs=self.fc,
                                      units=self.action_size,
                                      activation=None,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name='logits')

        with tf.name_scope('softmax'):
            self.action_distribution = tf.nn.softmax(self.logits)

        with tf.name_scope("loss"):
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.actions)
            self.loss = tf.reduce_mean(tf.reduce_sum(self.neg_log_prob * self.discounted_episode_rewards, axis=1) / self.length)

        with tf.name_scope("train"):
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    @staticmethod
    def conv_layer(inputs, filters, kernel_size, strides, name):
        with tf.variable_scope(name):
            conv = tf.layers.conv2d(inputs=inputs,
                                    filters=filters,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding='VALID',
                                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                    name='conv')

            conv_batchnorm = tf.layers.batch_normalization(conv,
                                                           training=True,
                                                           epsilon=1e-5,
                                                           name='batch_norm')
            conv_out = tf.nn.elu(conv_batchnorm, name='out')
        return conv_out
