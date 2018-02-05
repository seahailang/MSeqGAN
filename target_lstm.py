#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: target_lstm.py
@time: 2018/2/1 9:48
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from collections import namedtuple
import numpy as np

FLAGS = tf.app.flags.FLAGS


Params = namedtuple('params', ['w_r','b_r','w_z','b_z', 'w', 'b','dense_w','dense_b'])



class MGRUCell(object):
    def __init__(self,params):
        self.w_r = tf.convert_to_tensor(params.w_r,dtype=tf.float32)
        self.b_r = tf.convert_to_tensor(params.b_r,dtype=tf.float32)
        self.w_z = tf.convert_to_tensor(params.w_z,dtype=tf.float32)
        self.b_z = tf.convert_to_tensor(params.b_z,dtype=tf.float32)
        self.w = tf.convert_to_tensor(params.w,dtype=tf.float32)
        self.b = tf.convert_to_tensor(params.b,dtype=tf.float32)

    def __call__(self,inputs,states):
        r = tf.sigmoid(tf.matmul(inputs,self.w_r)+self.b_r)
        z = tf.sigmoid(tf.matmul(states,self.w_z)+self.b_z)
        r_states = r*states
        c = tf.tanh(tf.matmul(tf.concat([inputs,r_states],axis=-1),self.w)+self.b)
        new_states = (1-z)*states+z*c
        # new_states = states
        return new_states,new_states

class MDense(object):
    def __init__(self,params):
        self.w = tf.convert_to_tensor(params.dense_w,dtype=tf.float32)
        self.b = tf.convert_to_tensor(params.dense_b,dtype=tf.float32)
    def __call__(self,inputs):
        return tf.nn.softmax(tf.matmul(inputs,self.w)+self.b)




class TargetLSTM(object):
    def __init__(self,batch_size,max_len,embedding_matrix,params):
        output_size = embedding_matrix.shape[-1]
        self.cell = MGRUCell(params)
        self.dense = MDense(params)
        self.batch_size= batch_size
        self.max_len = max_len
        self.num_tokens= embedding_matrix.shape[0]
        self.start_token = tf.ones(shape=[self.batch_size],dtype=tf.int32)

        self.hidden_size = params.w.shape[-1]
        self.output_size = output_size
        self.embedding_matrix = tf.convert_to_tensor(embedding_matrix,dtype=tf.float32)
        #start with shape [batch_size,
        self.start = tf.nn.embedding_lookup(self.embedding_matrix,self.start_token)

    def loop_fn(self,i, output_ta, token_ta, state, embedding_matrix):
        new_input = output_ta.read(i)

        output, state = self.cell(new_input, state)
        output = self.dense(output)
        token_ta = token_ta.write(i, tf.cast(tf.argmax(output, axis=-1), tf.int32))
        i = i + 1
        output_ta = output_ta.write(i, tf.nn.embedding_lookup(embedding_matrix, tf.argmax(output, -1)))
        return i, output_ta, token_ta, state
    def gen_op(self):
        i = 0
        state = tf.random_uniform(shape=[self.batch_size,self.hidden_size])
        output_ta = tf.TensorArray(dtype=tf.float32,size=self.max_len+1,clear_after_read=False)
        output_ta = output_ta.write(0,self.start)
        token_ta = tf.TensorArray(dtype=tf.int32,size=self.max_len)

        def loop_fn(i, output_ta, token_ta, state, embedding_matrix):
            new_input = output_ta.read(i)

            output, state = self.cell(new_input, state)
            output = self.dense(output)
            token_ta = token_ta.write(i, tf.cast(tf.argmax(output, axis=-1), tf.int32))
            i = i + 1
            output_ta = output_ta.write(i, tf.nn.embedding_lookup(embedding_matrix, tf.argmax(output, -1)))
            return i, output_ta, token_ta, state

        i,output_ta,token_ta,final_state = tf.while_loop(
            cond = lambda a,b,c,d:tf.less(a,self.max_len),
            body = lambda a,b,c,d:loop_fn(a,b,c,d,self.embedding_matrix),
            loop_vars = [i,output_ta,token_ta,state]
        )
        print(token_ta)
        tokens= token_ta.stack()
        tokens = tf.transpose(tokens,[1,0])
        outputs = output_ta.stack()
        outputs = tf.transpose(outputs,[1,0,2])
        return tokens,outputs,final_state
    def log_likelihood(self,seqs):
        max_len = seqs.shape[-1]
        one_hot = tf.one_hot(seqs,depth=self.num_tokens)
        embedding = tf.nn.embedding_lookup(self.embedding_matrix,seqs)
        embedding = tf.transpose(embedding,[1,0,2])
        input_ta = tf.TensorArray(dtype=tf.float32,size=max_len)
        input_ta = input_ta.unstack(embedding)
        output_ta = tf.TensorArray(dtype=tf.float32,size=max_len)
        state = tf.zeros(dtype=tf.float32,shape=(self.batch_size,self.hidden_size))

        i = 0
        def loop_fn(cell,dense,i,output_ta,state,input_ta):
            inputs = input_ta.read(i)
            outputs,state = cell(inputs,state)
            outputs = dense(outputs)
            output_ta = output_ta.write(i,outputs)
            return i+1,output_ta,state
        i,output_ta,state = tf.while_loop(
            cond = lambda a,b,c:tf.less(a,max_len),
            body = lambda a,b,c:loop_fn(self.cell,self.dense,a,b,c,input_ta),
            loop_vars = [i,output_ta,state]
        )
        likelihoods = output_ta.stack()
        likelihoods = tf.transpose(likelihoods,[1,0,2])
        like = one_hot*likelihoods
        like = tf.reduce_sum(like,axis=-1)
        log_like = tf.reduce_sum(tf.log(like),axis=-1)
        return log_like



def test_gen():
    params = Params(w_r=np.random.uniform(-1,1,(200,100)).astype(np.float32),
                    b_r=np.zeros((100,)).astype(np.float32),
                    w_z = np.random.uniform(-1,1,(100,100)).astype(np.float32),
                    b_z = np.zeros((100,)).astype(np.float32),
                    w = np.random.uniform(-1,1,(200+100,100)).astype(np.float32),
                    b = np.zeros((100,)).astype(np.float32),
                    dense_w = np.random.uniform(-1,1,size=(100,200)),
                    dense_b = np.zeros(shape=(200,))
                    )
    embedding_matrix = np.random.uniform(low=-1,high=1,size=(200,200))

    t = TargetLSTM(10,20,embedding_matrix,params)
    gen_op = t.gen_op()
    with tf.Session() as sess:
        a = sess.run(gen_op)

        print(a[0])


def test_likely():
    params = Params(w_r=np.random.uniform(-1, 1, (200, 100)).astype(np.float32),
                    b_r=np.zeros((100,)).astype(np.float32),
                    w_z=np.random.uniform(-1, 1, (100, 100)).astype(np.float32),
                    b_z=np.zeros((100,)).astype(np.float32),
                    w=np.random.uniform(-1, 1, (200 + 100, 100)).astype(np.float32),
                    b=np.zeros((100,)).astype(np.float32),
                    dense_w=np.random.uniform(-1, 1, size=(100, 200)),
                    dense_b=np.zeros(shape=(200,))
                    )
    embedding_matrix = np.random.uniform(low=-1, high=1, size=(200, 200))
    seq = np.random.randint(200,size=(10,20))
    t = TargetLSTM(10, 20, embedding_matrix, params)
    like = t.log_likelihood(seq)
    with tf.Session() as sess:
        a = sess.run(like)
        print(a)



if __name__ == '__main__':
    test_likely()
