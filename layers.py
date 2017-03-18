import tensorflow as tf

def linear(input_tensor,output_size,scope,non_linearity=tf.nn.relu):
    input_size = input_tensor.get_shape().as_list()[1]
    with tf.variable_scope(scope):
        weights = tf.get_variable("weights",[input_size,output_size],tf.float32,initializer=tf.random_normal_initializer(stddev=0.002))
        bias = tf.get_variable("bias",[output_size],tf.float32,initializer=tf.constant_initializer(0))
        if(non_linearity==None):
            return tf.nn.bias_add(tf.matmul(input_tensor,weights),bias)
        else:
            return non_linearity(tf.nn.bias_add(tf.matmul(input_tensor,weights),bias))

def deconv2d(input_tensor,out_channels,scope,ksize=5,stride=2,non_linearity=tf.nn.relu):
    input_shape = input_tensor.get_shape().as_list()
    with tf.variable_scope(scope):
        filters = tf.get_variable("filters",[ksize,ksize,out_channels,input_shape[3]],tf.float32,initializer=tf.random_normal_initializer(stddev=0.002))
        biases = tf.get_variable("bias",[out_channels],tf.float32,initializer=tf.constant_initializer(0))
        deconv = tf.nn.conv2d_transpose(input_tensor,filters,output_shape=[input_shape[0],input_shape[1]*stride,input_shape[2]*stride,out_channels],strides=[1,stride,stride,1])
        deconv = tf.nn.bias_add(deconv,biases)
        return non_linearity(deconv)

def conv2d(input_tensor,out_channels,scope,ksize=5,stride=2,non_linearity=tf.nn.relu):
    input_shape = input_tensor.get_shape().as_list()
    with tf.variable_scope(scope):
        filters = tf.get_variable("filters",[ksize,ksize,input_shape[3],out_channels],tf.float32,initializer=tf.random_normal_initializer(stddev=0.002))
        biases = tf.get_variable("bias",[out_channels],tf.float32,initializer=tf.constant_initializer(0))
        conv = tf.nn.conv2d(input_tensor,filters,strides=[1,stride,stride,1], padding='SAME')
        conv = tf.nn.bias_add(conv,biases)
        return non_linearity(conv)
