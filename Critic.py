import tensorflow as tf
from layers import *

class Critic(object):
    def __init__(self,model_name,batch_size=32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.has_built = False

    def build(self,input_images):
        with tf.variable_scope("Critic_"+self.model_name,reuse=self.has_built):
            self.h0 = conv2d(input_images,128,"h0")
            self.h1 = conv2d(self.h0,256,"h1")
            self.h2 = conv2d(self.h1,512,"h2")
            self.h3 = conv2d(self.h2,1024,"h3")
            #self.h3_flat = tf.reshape(self.h3,[self.batch_size,1024*4*4])
            #self.out = linear(self.h3_flat,1,"out",non_linearity=None)
            self.out = tf.reduce_mean(self.h3)
            self.has_built=True

            critic_vars = [v for v in tf.trainable_variables() if v.name.startswith("Critic_"+self.model_name)]
            self.clip_op = [var.assign(tf.clip_by_value(var,-0.01,0.01)) for var in critic_vars]

            return self.out
