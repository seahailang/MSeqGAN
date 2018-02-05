#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: generator.py
@time: 2018/2/1 19:57
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

class Generator(object):
    def __init__(self,units,batch_size,num_token):
        # assert len(start_token) == batch_size, 'start_token numbers should be euqal with batch size'
        # cells = [tf.nn.rnn_cell.GRUCell(unit) for unit in units]
        self.units = units
        self.num_token = num_token
        self.cell = tf.nn.rnn_cell.GRUCell(self.units)
        self.dense = tf.layers.Dense(units=num_token)
        self.embedding = tf.get_variable('embedding',
                                         shape = [],
                                         dtype = tf.float32)
        self.batch_size = batch_size
        # self.start_token = start_token
        self.zero_state = tf.zeros(shape=[batch_size,units],dtype=tf.float32)
        self.trainable_layer = [self.cell,self.dense]

    # def get_start(self):
    #     return tf.nn.embedding_lookup(self.embedding,self.start_token)
    def embedding_op(self,tokens):
        return tf.nn.embedding_lookup(self.embedding,tokens)
    def dynamic_decode(self,start,state,length=20):
        output_ta = tf.TensorArray(dtype=tf.float32,size=length+1,clear_after_read=False)
        output_ta = output_ta.write(0,start)
        token_ta = tf.TensorArray(dtype=tf.int32,size=length)
        i = 0
        def loop_fn(i,output_ta,token_ta,state):
            inputs = output_ta.read(i)
            output,state = self.cell(inputs,state)
            output = self.dense(output)
            token= tf.argmax(output,axis=-1)
            token_ta.write(i,token)
            output = tf.nn.embedding_lookup(self.embedding,token)
            i = i+1

            output_ta = output_ta.write(i,output)
            return i,output_ta,token_ta,state
        _,output_ta,token_ta, _ = tf.while_loop(cond = lambda a,b,c,d :tf.less(a,length),
                                       body = loop_fn,
                                       loop_vars = [i,output_ta,token_ta,state],
                                       parallel_iterations = self.batch_size)
        outputs = output_ta.stack()
        logits = tf.slice(outputs,[1,0,0],size=[length,self.batch_size,self.num_token])
        logits = tf.transpose(logits,[1,0,2])
        tokens = token_ta.stack()
        tokens = tf.transpose(tokens,[1,0,2])
        return logits,tokens
    def __call__(self, start,state,length):
        return self.dynamic_decode(start,state,length)

    def decode(self,state,token):
        inputs = tf.nn.embedding_lookup(self.embedding,token)
        logits,state = self.cell(inputs,state)
        logits = self.dense(logits)
        logits = tf.nn.softmax(logits)
        return logits,state

    def encode(self,seq,init_state=None):
        if init_state:
            state  = init_state
        else:
            state = tf.zeros(shape=[self.batch_size,self.units])
        seq_len =seq.shape[-1]
        seq_embedded = tf.nn.embedding_lookup(self.embedding,seq)
        seq_embedded = tf.transpose(seq_embedded,[1,0,2])
        inputs_ta = tf.TensorArray(dtype=tf.float32).unstack(seq_embedded)

        def loop_fn(i,state,inputs_ta):
            output,state = self.cell(inputs_ta.read(i),state)
            i = i+1
            return i,state,output
        i = 0
        _,state,output = tf.while_loop(cond = lambda a,b: tf.less(a,seq_len),
                                body = lambda a,b:loop_fn(a,b,inputs_ta),
                                loop_vars = [i,state])
        output = self.dense(output)
        output = tf.argmax(output,axis=-1)
        return output,init_state

    def trainable_variables(self):
        variables = []
        for i in range(len(self.trainable_layer)):
            variables.extend(self.trainable_layer[i].trainable_variables())
        return variables




if __name__ == '__main__':
    pass