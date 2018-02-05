#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: rollout.py
@time: 2018/2/5 9:52
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

class RollOut(object):
    def __init__(self,roll_nums):
        self.roll_nums = roll_nums
        # self.random_seed = seed
    def rewards(self,seqs,generator,discriminator):
        length = seqs.shape[-1]
        seqs = tf.transpose(seqs,[1,0])
        input_ta = tf.TensorArray(dtype=tf.int32, size=length, clear_after_read=False)
        input_ta = input_ta.unstack(generator.embedding_op(seqs))
        rewards = []
        for i in range(len(self.roll_nums)):
            reward = []
            output_ta = tf.TensorArray(dtype=tf.int32,size=length,clear_after_read=False)
            output_ta = output_ta.write(input_ta.read(0))

            def loop_fn(roll_times,current_times,input_ta,output_ta,state):
                inputs = output_ta.read(roll_times)
                inputs = generator.embedding_op(inputs)
                logits,state = generator.decode(inputs,state)
                roll_times = roll_times + 1
                output = tf.cond(roll_times<current_times,
                                 lambda:input_ta.read(roll_times),
                                 lambda:tf.multinomial(logits,num_samples=1,output_dtype=tf.int32))
                output_ta = output_ta.write(roll_times,output)
                return roll_times,output_ta,state

            for j in range(1,length):
                roll_times =j
                state = generator.zero_state
                _,output_ta,_ = tf.while_loop(cond=lambda a,b,c:tf.less(a,length),
                                              body=lambda a,b,c:loop_fn(a,j,input_ta,b,c),
                                              loop_vars = [roll_times,output_ta,state])
                outputs = output_ta.stack()
                outputs = tf.transpose(outputs,[1,0])
                reward.append(discriminator(outputs))
            reward = tf.concat(reward,axis=1)
            reward = tf.reduce_mean(reward)
            rewards.append(reward)
            rewards.append(discriminator(seqs))
        rewards = tf.concat(rewards,axis=-1)

        return rewards












if __name__ == '__main__':
    pass