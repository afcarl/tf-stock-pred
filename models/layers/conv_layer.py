import tensorflow as tf
import utils.summarizer as s





def gated_conv2d(x, filter_size, in_channel, out_channel, strides=[1, 1, 1, 1], padding="SAME", name="residual_cnn"):
    with tf.variable_scope(name) as vs:
        filter_shape = filter_size + [in_channel, out_channel]
        # variable definition
        W = tf.get_variable('weight_filter', shape=filter_shape,
                            initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            regularizer=None)

        b = tf.get_variable('bias_filter', shape=[out_channel])

        W_t = tf.get_variable('weight_gate', shape=filter_shape,
                              initializer=tf.contrib.layers.xavier_initializer_conv2d())

        b_t = tf.get_variable('bias_gate', shape=out_channel)

        # convolution
        conv_filter = tf.nn.conv2d(x, W, strides, padding)
        conv_gate = tf.nn.conv2d(x, W_t, strides, padding)

        conv_filter = tf.add(conv_filter, b)
        conv_gate = tf.add(conv_gate, b_t)

        # gates
        H = tf.tanh(conv_filter, name='activation')
        T = tf.sigmoid(conv_gate, name='transform_gate')

        # debugging
        tf.summary.histogram(vs.name + "_weight_filter", W)
        tf.summary.histogram(vs.name + '_bias_filter', b)
        tf.summary.histogram(vs.name + '_weight_gate', W_t)
        tf.summary.histogram(vs.name + '_bias_gate', b_t)
        s._norm_summary(W, vs.name)
        s._norm_summary(W_t, vs.name)

        return tf.multiply(H, T)




def highway_conv2d(x, filter_size, in_channel, out_channel, strides=[1, 1, 1, 1], padding="SAME", name="highway_cnn"):
    with tf.variable_scope(name) as vs:
        filter_shape = filter_size + [in_channel, out_channel]
        # variable definition
        W = tf.get_variable('weight_filter', shape=filter_shape,
                            initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            regularizer=None)

        b = tf.get_variable('bias_filter', shape=[out_channel])

        W_t = tf.get_variable('weight_gate', shape=filter_shape,
                              initializer=tf.contrib.layers.xavier_initializer_conv2d())

        b_t = tf.get_variable('bias_gate', shape=out_channel,
                              initializer=tf.constant_initializer(-3.))

        # convolution
        conv_filter = tf.nn.conv2d(x, W, strides, padding)
        conv_gate = tf.nn.conv2d(x, W_t, strides, padding)

        conv_filter = tf.add(conv_filter, b)
        conv_gate = tf.add(conv_gate, b_t)

        # gates
        H = tf.tanh(conv_filter, name='activation')
        T = tf.sigmoid(conv_gate, name='transform_gate')
        C = tf.subtract(1.0, T, name='carry_gate')

        # debugging
        tf.summary.histogram(vs.name + "_weight_filter", W)
        tf.summary.histogram(vs.name + '_bias_filter', b)
        tf.summary.histogram(vs.name + '_weight_gate', W_t)
        tf.summary.histogram(vs.name + '_bias_gate', b_t)
        s._norm_summary(W, vs.name)
        s._norm_summary(W_t, vs.name)

        return tf.add(tf.multiply(H, T), tf.multiply(x, C))

def conv2d(x, filter_size, in_channel, out_channel, strides=[1,1,1,1], padding="VALID", name="cnn", activation_fn=tf.tanh):
    with tf.variable_scope(name) as vs:
        filter_shape = filter_size + [in_channel, out_channel]
        W = tf.get_variable('kernel', shape=filter_shape)
        b = tf.get_variable('bias', shape=out_channel)

        activation = tf.add(tf.nn.conv2d(x, W, strides, padding), b)
        if activation_fn:
            activation = activation_fn(activation)

        tf.summary.histogram(vs.name + '_filter', W)
        tf.summary.histogram(vs.name + '_biases_filter', b)
        s._norm_summary(W, vs.name)
    return activation