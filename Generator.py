import tensorflow as tf
from layers import *

class Generator(object):
    def __init__(self,model_name,batch_size=32,z_dim=100,output_x=32,output_y=32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.z = tf.placeholder(tf.float32,[batch_size,z_dim])
        self.output_x = output_x
        self.output_y = output_y

    def build(self):
        with tf.variable_scope("Generator_" + self.model_name) as scope:
            self.upsampled = linear(self.z,4*4*1024,"upsample",non_linearity=None)
            self.reshaped = tf.reshape(self.upsampled,[self.batch_size,4,4,1024])
            self.h0 = deconv2d(self.reshaped,512,"h0")
            self.h1 = deconv2d(self.h0,256,"h1")
            self.h2 = deconv2d(self.h1,128,"h2")
            self.h3 = deconv2d(self.h2,3,"h3",non_linearity=tf.nn.tanh)

            return self.h3
