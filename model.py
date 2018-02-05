#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: model.py
@time: 2018/2/5 10:52
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from discriminator import Discriminator
from generator import Generator
from rollout import RollOut
from dataset import dataset

FLAGS = tf.app.flags.FLAGS


class GAN(object):
    def __init__(self,generator,discriminator,rollout,
                 init_learning_rate=FLAGS.learning_rate,
                 decay_steps=FLAGS.decay_steps,
                 decay_rate=FLAGS.decay_rate,
                 seq_len=FLAGS.seq_len,
                 lr_decay = FLAGS.lr_decay):
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = generator.batch_size
        self.units = generator.units
        self.rollout = rollout
        self.init_learning_rate = init_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.seq_len = seq_len
        self.lr_decay = lr_decay
        self.num_token = generator.num_token
        self.rewards_placeholder = tf.placeholder(dtype=tf.float32,shape=[self.batch_size,self.seq_len])

    def learing_rate(self,global_step):
        if self.lr_decay:

            return tf.train.exponential_decay(learning_rate=self.init_learning_rate,
                                              global_step=global_step,
                                              decay_steps=self.decay_steps,
                                              decay_rate=self.decay_rate)
        else:
            return self.init_learning_rate


    def gen(self,init_state=None,start_token = None):
        if not init_state:
            init_state = tf.random_uniform(shape=[self.batch_size,self.units],minval=-1,maxval=1,dtype=tf.float32)
        if not start_token:
            start_token = tf.ones(shape=[self.batch_size],dtype=tf.int32)
        start = tf.nn.embedding_lookup(self.generator.embedding,start_token)
        gen_op = self.generator(init_state,start,length=self.seq_len)
        return gen_op

    def compute_rewards(self,seqs):
        return self.rollout.rewards(seqs,self.generator,self.discriminator)

    def gen_loss_op(self,seq,logits,rewards):
        seq = tf.convert_to_tensor(seq,dtype=tf.int32)
        loss = tf.losses.softmax_cross_entropy(seq,logits)
        loss = loss*rewards
        return loss
    def dsc_loss_op(self,labels,logits):
        loss = tf.losses.log_loss(labels,logits)
        return loss

    def gen_train_op(self,loss,lr):
        opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
        vars = self.generator.trainable_variables()
        grads_and_vars = opt.compute_gradients(loss,var_list = vars)
        op = opt.apply_gradients(grads_and_vars)
        return op

    def dsc_train_op(self,loss,lr):
        opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
        vars = self.discriminator.trainable_variables()
        grads_and_vars = opt.compute_gradients(loss, var_list=vars)
        op = opt.apply_gradients(grads_and_vars)
        return op


def train(gan,train_iterator):

    iterator_placeholder = tf.placeholder(dtype=tf.string,shape=[])
    iterator = tf.data.Iterator.from_string_handle(iterator_placeholder,
                                                   output_shapes=train_iterator.output_shapes,
                                                   output_types=train_iterator.output_types)
    real = iterator.get_next()

    # here is op for generator, the rewards_op should not update by gradient
    # so we should first run the rewards op and feed the results in train_op
    random_state = tf.placeholder(shape=gan.generator.zero_state.shape,dtype=tf.float32)
    logits,fake = gan.gen(initial_state = random_state)
    rewards_op = gan.compute_rewards(fake)


    # add placeholder to get the result of generator
    # the gradients can't pass the placeholder
    # fake = tf.placeholder(shape=[gan.batch_size,gan.seq_len],dtype=tf.int32)
    # logits = tf.placeholder(shape=[gan.batch_size,gan.seq_len,gan.num_token],dtype=tf.float32)
    rewards = gan.rewards_placeholder
    gen_loss = gan.gen_loss_op(fake,logits,rewards)
    fake_logits = gan.discriminator(fake)
    real_logits = gan.discriminator(real)
    fake_loss = gan.dsc_loss_op(0,fake_logits)
    real_loss = gan.dsc_loss_op(1,real_logits)
    global_step = tf.train.create_global_step()
    dsc_loss = 0.5*fake_loss+0.5*real_loss
    learning_rate = gan.learing_rate(global_step)
    gen_op = gan.gen_train_op(gen_loss,learning_rate)
    dsc_op = gan.dsc_train_op(dsc_loss,learning_rate)
    with tf.Session() as sess:
        handle = sess.run(train_iterator.string_handle())
        feeds ={iterator_placeholder:handle}
        for i in range(FLAGS.max_train_steps):
            state = np.random.uniform(low=-1,high=1,size=gan.generator.zero_state.shape)
            feeds[random_state]=state
            rewards_array = sess.run([rewards_op],feed_dict=feeds)

            feeds[rewards] = rewards_array
            sess.run([gen_op,dsc_op],feed_dict=feeds)


def main():
    generator = Generator(units=100,
                          batch_size = FLAGS.batch_size,
                          num_token=FLAGS.num_token)
    discriminator = Discriminator()

    rollout = RollOut(roll_nums=5)

    gan = GAN(generator =generator,discriminator=discriminator,rollout=rollout)
    iterator = dataset.make_one_shot_iterator()
    train(gan,iterator)











if __name__ == '__main__':
    main()