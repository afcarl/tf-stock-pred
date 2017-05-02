import tensorflow as tf
from tensorflow.python.ops import nn, init_ops, standard_ops
from tensorflow.contrib.layers.python.layers import initializers
import utils.summarizer as s

def gated_dense_layer_over_time(x, in_size, out_size, sequence_length, scope_name, activation_fn=tf.nn.elu):
    layers_output = []
    with tf.variable_scope(scope_name) as vs:
        W = tf.get_variable('weight_filter', shape=[in_size, out_size],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=None)

        b = tf.get_variable('bias_filter',
                            shape=[out_size],
                            initializer=tf.constant_initializer(0.))

        W_t = tf.get_variable('weight_gate', shape=[in_size, out_size],
                              initializer=tf.contrib.layers.xavier_initializer_conv2d())

        b_t = tf.get_variable('bias_gate',
                              shape=[out_size],
                              initializer=tf.constant_initializer(0.))

        # Iterate over the timestamp
        for t in range(0, sequence_length):

            H = activation_fn(tf.add(tf.matmul(x[:, t, :], W), b), name="activation")
            T = tf.sigmoid(tf.add(tf.matmul(x[:, t, :], W_t), b_t), name="transit_gate")
            layer_output = tf.multiply(H, T)

            # apply dropout
            # if h_params.dropout is not None and mode == tf.contrib.learn.ModeKeys.TRAIN:
            #     layer_output = tf.nn.dropout(layer_output, keep_prob=1 - h_params.dropout)

            layers_output.append(tf.expand_dims(layer_output, 1))  # add again the timestemp dimention to allow concatenation
        # proved to be the same weights
        s.add_hidden_layer_summary(layers_output[-1], vs.name)

        tf.summary.histogram(vs.name + "_weight_filter", W)
        tf.summary.histogram(vs.name + '_bias_filter', b)
        tf.summary.histogram(vs.name + '_weight_gate', W_t)
        tf.summary.histogram(vs.name + '_bias_gate', b_t)
        s._norm_summary(W, vs.name)
        s._norm_summary(W_t, vs.name)
    return tf.concat(layers_output, axis=1)

def highway_dense_layer_over_time(x, in_size, out_size, sequence_length, scope_name, activation_fn=tf.nn.elu, init_bias=-3.):
    layers_output = []
    with tf.variable_scope(scope_name) as vs:
        W = tf.get_variable('weight_filter', shape=[in_size, out_size],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=None)

        b = tf.get_variable('bias_filter', shape=[out_size],
                            initializer=tf.constant_initializer(0.))

        W_t = tf.get_variable('weight_gate', shape=[in_size, out_size],
                              initializer=tf.contrib.layers.xavier_initializer_conv2d())

        b_t = tf.get_variable('bias_gate', shape=[out_size],
                              initializer=tf.constant_initializer(init_bias)
                              )

        # Iterate over the timestamp
        for t in range(0, sequence_length):

            H = activation_fn(tf.add(tf.matmul(x[:, t, :], W), b), name="activation")
            T = tf.sigmoid(tf.add(tf.matmul(x[:, t, :], W_t), b_t), name="transit_gate")
            C = tf.subtract(1.0, T, name='carry_gate')
            layer_output = tf.add(tf.multiply(H, T), tf.multiply(x[:, t, :], C))
            # apply dropout
            # if h_params.dropout is not None and mode == tf.contrib.learn.ModeKeys.TRAIN:
            #     layer_output = tf.nn.dropout(layer_output, keep_prob=1 - h_params.dropout)

            layers_output.append(tf.expand_dims(layer_output, 1))  # add again the timestemp dimention to allow concatenation
        # proved to be the same weights
        s.add_hidden_layer_summary(layers_output[-1], vs.name)

        tf.summary.histogram(vs.name + "_weight_filter", W)
        tf.summary.histogram(vs.name + '_bias_filter', b)
        tf.summary.histogram(vs.name + '_weight_gate', W_t)
        tf.summary.histogram(vs.name + '_bias_gate', b_t)
        s._norm_summary(W, vs.name)
        s._norm_summary(W_t, vs.name)
    return tf.concat(layers_output, axis=1)

# apply linera transformation to reduce the dimension


def dense_layer_over_time_std(x, in_size, out_size, sequence_length, scope_name, activation_fn=tf.nn.elu):
    layers_output = []
    with tf.variable_scope(scope_name, reuse=True) as vs:
        for t in range(0, sequence_length):
            layer_output = tf.contrib.layers.fully_connected(inputs=x[:, t, :],
                                              num_outputs=out_size,
                                              activation_fn=activation_fn,
                                              scope=vs)

            layers_output.append(tf.expand_dims(layer_output, 1))  # add again the timestemp dimention to allow concatenation
        # proved to be the same weights
        s.add_hidden_layer_summary(layers_output[-1], vs.name, weight=tf.get_variable("weights"))
        tf.summary.histogram(vs.name + '_bias', tf.get_variable("biases"))

    return tf.concat(layers_output, axis=1)



def dense_layer_over_time(x, in_size, out_size, sequence_length, scope_name, activation_fn=tf.nn.elu):
    layers_output = []
    with tf.variable_scope(scope_name) as vs:
        W = tf.get_variable('weight_filter', shape=[in_size, out_size],
                            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable('bias_filter', shape=[out_size],
                            initializer=tf.constant_initializer(0.))


        for t in range(0, sequence_length):
            layer_output = standard_ops.add(standard_ops.matmul(x[:, t, :], W), b)
            layer_output = activation_fn(layer_output)

            layers_output.append(tf.expand_dims(layer_output, 1))  # add again the timestemp dimention to allow concatenation
        # proved to be the same weights
        s.add_hidden_layer_summary(layers_output[-1], vs.name, weight=W)
        tf.summary.histogram(vs.name + '_bias', b)

    return tf.concat(layers_output, axis=1)


class Dense(object):
    """Densely-connected layer class.
    This layer implements the operation:
    `outputs = activation(inputs.kernel + bias)`
    Where `activation` is the activation function passed as the `activation`
    argument (if not `None`), `kernel` is a weights matrix created by the layer,
    and `bias` is a bias vector created by the layer
    (only if `use_bias` is `True`).
    Note: if the input to the layer has a rank greater than 2, then it is
    flattened prior to the initial matrix multiply by `kernel`."""

    def __init__(self, n_in, n_out, vs,
                 activation_fn=tf.nn.relu,
                 weights_initializer=initializers.xavier_initializer,
                 weights_regularizer=None,
                 bias_initializer=init_ops.zeros_initializer(),
                 normalizer_fn=None,
               ):

        if normalizer_fn and not bias_initializer:
            self.use_bias = False
        elif not normalizer_fn and bias_initializer:
            self.use_bias = True
        else:
            raise Exception("Need bias")

        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn

        self.W = vs.get_variable('weights',
                                 shape=[self.n_in, self.n_out],
                                 initializer=weights_initializer(),
                                 regularizer=weights_regularizer,
                                 trainable=True)
        if self.use_bias:
            self.b = vs.get_variable('bias',
                                     shape=[self.n_out, ],
                                     initializer=bias_initializer(),
                                     trainable=True)
            self.normalizer_fn=None

        else:
            self.bias = None
            self.normalizer_fn=normalizer_fn

    def call(self, inputs):
        outputs = standard_ops.matmul(inputs, self.W)

        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)

        # Apply normalizer function / layer.
        if self.normalizer_fn is not None:
            outputs = self.normalizer_fn(outputs)

        if self.activation_fn is not None:
            outputs = self.activation_fn(outputs)  # pylint: disable=not-callable

        return outputs