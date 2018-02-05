#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: discriminator.py
@time: 2018/2/1 19:39
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


class Discriminator(object):
    def __init__(self):
        self.cnn_layer1 = tf.layers.Conv1D(filters=100,kernel_size=3,strides=1,padding='same')
        self.pool_layer1 = tf.layers.MaxPooling1D(pool_size=3,strides=2,padding='same')
        self.cnn_layer2 = tf.layers.Conv1D(filters=32,kernel_size=3,strides=1,padding='same')
        self.pool_layer2 = tf.layers.MaxPooling1D(pool_size=3,strides=2,padding='same')
        self.pool_over_time = tf.layers.MaxPooling1D(pool_size=5,strides=5,padding='same')
        self.dense1 = tf.layers.Dense(100,activation=tf.nn.relu)
        self.softmax = tf.layers.Dense(1)
        self.params_layers = [self.cnn_layer1,self.cnn_layer2,self.dense1,self.softmax]

    def __call__(self,input):
        with tf.variable_scope('Discriminator'):
            c1 = self.cnn_layer1(input)
            p1 = self.pool_layer1(c1)
            c2 = self.cnn_layer2(p1)
            p2 = self.pool_layer2(c2)
            p = tf.transpose(p2,[0,2,1])
            p = self.pool_over_time(p)
            p = tf.reshape(p,[p.shape[0],-1])
            d1 = self.dense1(p)
            logits = self.softmax(d1)
        return logits

    def trainable_variables(self):
        variables = []
        for i in range(len(self.params_layers)):
            variables.extend(self.params_layers[i].trainable_variables())
        return variables



if __name__ == '__main__':
    pass