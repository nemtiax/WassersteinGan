from glob import glob
from utilities import *

from Critic import Critic
from Generator import Generator
from Pair import Pair

import tensorflow as tf
import os
import numpy as np

class Trainer(object):
    def __init__(self,batch_size=64):
        self.batch_size = batch_size

    def load_data(self,path):
        self.data_files = glob(os.path.join(path, "*.jpg"))
        self.data = [load_image(data_file,64) for data_file in self.data_files]

    def get_batch(self,index):
        batch_images = self.data[self.batch_size*index:self.batch_size*(index+1)]
        batch_z = np.random.uniform(-1, 1, size=(self.batch_size,100))

        return batch_images,batch_z

    def train(self,pair,epochs,sess):
        num_batches = len(self.data)//self.batch_size

        sample_z = np.random.uniform(-1, 1, size=(self.batch_size,100))
        sample_batch,_ = self.get_batch(0)
        sample_batch = np.copy(sample_batch)
        count = 0

        gen_train = pair.generator_train_op
        critic_train = pair.critic_train_op
        critic_clip = pair.critic.clip_op

        for ep in range(epochs):
            for batch_index in range(num_batches):
                batch_images,batch_z = self.get_batch(batch_index)

                sess.run(gen_train,feed_dict={pair.z: batch_z})
                sess.run(critic_train,feed_dict={pair.z: batch_z,pair.real_images: batch_images})
                sess.run(critic_clip)
                count = count+1
                if(count%10==0):
                    print("Epoch %02d, Batch %04d"%(ep,batch_index))
                    generated_sample = sess.run(pair.generator_out,feed_dict={pair.z: sample_z})
                    g_loss,c_loss = sess.run([pair.generator_loss,pair.critic_loss],feed_dict={pair.z: batch_z,pair.real_images: batch_images})
                    #print(generated_sample)
                    #print(batch_images)
                    #first_layer = sess.run(pair.generator.upsampled,feed_dict={pair.z: sample_z})
                    #print(first_layer)
                    #second_layer = sess.run(pair.generator.h0,feed_dict={pair.z: sample_z})
                    #print(second_layer)
                    print(np.mean(generated_sample))
                    print(np.mean(batch_images))
                    print(g_loss)
                    print(c_loss)
                    save_mosaic(generated_sample,"samples/sample_%02d_%04d.png"%(ep,batch_index))

with tf.Session() as sess:
    g = Generator("g1")
    c = Critic("c1")
    p = Pair(g,c)
    p.build()
    tf.global_variables_initializer().run()

    trainer = Trainer()
    trainer.load_data('./celebA')
    trainer.train(p,10,sess)
