import tensorflow as tf
from layers import *

from Critic import Critic
from Generator import Generator

class Pair(object):
    def __init__(self,generator,critic):
        self.generator = generator
        self.z = self.generator.z
        self.critic = critic
        self.real_images = tf.placeholder(tf.float32,[64,64,64,3])

    def build(self):
        self.generator_out = self.generator.build()
        self.critic_real_out = self.critic.build(self.real_images)
        self.critic_fake_out = self.critic.build(self.generator_out)

        self.critic_loss = tf.reduce_mean(self.critic_real_out - self.critic_fake_out)
        self.generator_loss = tf.reduce_mean(self.critic_fake_out)

        all_vars = tf.trainable_variables()

        self.c_vars = [var for var in all_vars if 'Critic' in var.name]
        self.g_vars = [var for var in all_vars if 'Generator' in var.name]


        self.critic_train_op = tf.train.RMSPropOptimizer(2e-4,0.9).minimize(self.critic_loss,var_list=self.c_vars)
        self.generator_train_op = tf.train.RMSPropOptimizer(2e-4,0.9).minimize(self.generator_loss,var_list=self.g_vars)
