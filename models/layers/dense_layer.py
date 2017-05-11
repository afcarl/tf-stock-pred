import tensorflow as tf
from tensorflow import GraphKeys
from tensorflow.python.ops import nn, init_ops, standard_ops
from tensorflow.contrib.layers.python.layers import initializers
import utils.summarizer as s
import utils.func_utils as fu




def highway_dense_layer_ot(x, in_size, out_size, sequence_length, scope_name,
                                  activation_fn=tf.nn.elu,
                                  batch_norm=fu.create_BNParams()
                                  ):
    with tf.variable_scope(scope_name) as vs:
        x = dense_layer_ot(x, in_size, out_size, sequence_length, 'sub_1', activation_fn, batch_norm)
        x = gated_dense_layer_ot(x, in_size, out_size, sequence_length, 'sub_2', activation_fn, batch_norm, is_highway=True)

    return x


def gated_res_net_layer_ot(x, in_size, out_size, sequence_length, scope_name,
                        activation_fn=tf.nn.elu,
                        batch_norm=fu.create_BNParams()):
    orig_x = x

    with tf.variable_scope(scope_name) as vs:
        x = dense_layer_ot(x, in_size, out_size, sequence_length, 'sub_1', activation_fn, batch_norm)
        x = gated_dense_layer_ot(x, in_size, out_size, sequence_length, 'sub_2', activation_fn, batch_norm)

    with tf.variable_scope('sub_add'):
        x += orig_x

    return x


def gated_dense_layer_ot(x, in_size, out_size, sequence_length, scope_name,
                         activation_fn=tf.nn.elu,
                         batch_norm=fu.create_BNParams(),
                         is_highway=False):
    '''
    Apply a gated liner layer to the input.
    activattion_fn(mul(x,W)) * sigmoid(mul(x,W_t))
    The gate W_T should learn how much to filter the input x
    :param x: input mini-batch
    :param in_size: input size or feature number
    :param out_size: output size -> respect to a highway net can be what I what
    :param sequence_length: timestamp number
    :param scope_name: name of the scope of this layer
    :param activation_fn: activation function to apply
    :param batch_norm: apply batch norm before computing the activation of both W and W_t
    '''
    layers_output = []
    with tf.variable_scope(scope_name) as vs:
        W = tf.get_variable('weight_filter', shape=[in_size, out_size],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            collections=[GraphKeys.WEIGHTS, GraphKeys.GLOBAL_VARIABLES],
                            trainable=True)

        W_t = tf.get_variable('weight_gate', shape=[in_size, out_size],
                              initializer=tf.contrib.layers.xavier_initializer(),
                              collections=[GraphKeys.WEIGHTS, GraphKeys.GLOBAL_VARIABLES],
                              trainable=True)

        if not batch_norm.apply:
            b_t = tf.get_variable('bias_gate',
                                  shape=[out_size],
                                  initializer=tf.constant_initializer(0.),
                                  collections=[tf.GraphKeys.BIASES, GraphKeys.GLOBAL_VARIABLES])

            b = tf.get_variable('bias_filter',
                                shape=[out_size],
                                initializer=tf.constant_initializer(0.),
                                collections=[tf.GraphKeys.BIASES, GraphKeys.GLOBAL_VARIABLES])


        # Iterate over the timestamp
        for t in range(0, sequence_length):
            H_linear = tf.matmul(x[:, t, :], W)
            T_linear = tf.matmul(x[:, t, :], W_t)

            if batch_norm.apply:
                H_norm = tf.contrib.layers.batch_norm(H_linear,
                                                 center=batch_norm.center,
                                                 scale=batch_norm.scale,
                                                 is_training=batch_norm.phase,
                                                 scope=vs.name + '_filter_bn')
                H = activation_fn(H_norm, name="activation")


                T_norm = tf.contrib.layers.batch_norm(H_linear,
                                                 center=batch_norm.center,
                                                 scale=batch_norm.scale,
                                                 is_training=batch_norm.phase,
                                                 scope=vs.name + '_gate_bn')
                T = tf.sigmoid(T_norm, name="transit_gate")

            else:
                H = activation_fn(tf.add(H_linear, b), name="activation")
                T = tf.sigmoid(tf.add(T_linear, b_t), name="transit_gate")

            if is_highway:
                C = 1 - T
                layer_output = tf.multiply(H, T) + (C * x[:, t, :])
            else:
                layer_output = tf.multiply(H, T)
            # apply dropout
            # if h_params.dropout is not None and mode == tf.contrib.learn.ModeKeys.TRAIN:
            #     layer_output = tf.nn.dropout(layer_output, keep_prob=1 - h_params.dropout)

            layers_output.append(tf.expand_dims(layer_output, 1))  # add again the timestemp dimention to allow concatenation
        # proved to be the same weights
        s.add_hidden_layer_summary(layers_output[-1], vs.name)

        tf.summary.histogram(vs.name + "_weight_filter", W)
        tf.summary.histogram(vs.name + '_weight_gate', W_t)
        if not batch_norm.apply:
            tf.summary.histogram(vs.name + '_bias_filter', b)
            tf.summary.histogram(vs.name + '_bias_gate', b_t)

        s._norm_summary(W, vs.name + '_filter')
        s._norm_summary(W_t, vs.name + '_gate')

    return tf.concat(layers_output, axis=1)





def dense_layer_ot(x, in_size, out_size, sequence_length, scope_name,
                          activation_fn=tf.nn.elu,
                          batch_norm=fu.create_BNParams()
                          ):
    '''
    Apply a dense layer over all the time_stamp.
    This is for filtering the timeseries
    :param x: input data
    :param in_size: input size or number of feature
    :param out_size: output size
    :param sequence_length: length of the sequence. Number of timestemp to iterate of
    :param scope_name: scope name of this transformation
    :param activation_fn: activation function
    :param batch_norm: named indicating if applying batch normalization and the phase(true if training, false if tensing)
    :return: 
    '''
    layers_output = []
    with tf.variable_scope(scope_name) as vs:
        W = tf.get_variable('weight_filter', shape=[in_size, out_size],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            collections=[GraphKeys.WEIGHTS, GraphKeys.GLOBAL_VARIABLES],
                            trainable=True
                            )

        if not batch_norm.apply:
            b = tf.get_variable('bias_filter', shape=[out_size],
                                initializer=tf.constant_initializer(0.),
                                collections=[GraphKeys.BIASES, GraphKeys.GLOBAL_VARIABLES],
                                trainable=True
                                )


        for t in range(0, sequence_length):
            layer_output = standard_ops.matmul(x[:, t, :], W)

            if batch_norm.apply:
                layer_output = tf.contrib.layers.batch_norm(layer_output,
                                                            center=batch_norm.center,
                                                            scale=batch_norm.scale,
                                                            is_training=batch_norm.phase,
                                                            scope=vs.name + '_bn')
            else:
                # apply batch norm
                layer_output = standard_ops.add(layer_output, b)

            if activation_fn:
                layer_output = activation_fn(layer_output)

            layers_output.append(tf.expand_dims(layer_output, 1))  # add again the timestemp dimention to allow concatenation

        # proved to be the same weights
        s.add_hidden_layer_summary(layers_output[-1], vs.name, weight=W)
        if not batch_norm.apply:
            tf.summary.histogram(vs.name + '_bias', b)

    return tf.concat(layers_output, axis=1)


def dense_layer(x, in_size, out_size, scope,
                activation_fn=tf.nn.elu):

    dense_layer = Dense(in_size, out_size, scope, activation_fn=activation_fn)

    return dense_layer.call(x)

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
                 normalizer_fn=None,
               ):

        if normalizer_fn:
            self.use_bias = False
        else:
            self.use_bias = True

        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn

        self.W = tf.get_variable('weights',
                                 shape=[self.n_in, self.n_out],
                                 initializer=weights_initializer(),
                                 regularizer=weights_regularizer,
                                 trainable=True,
                                 collections=[tf.GraphKeys.WEIGHTS, GraphKeys.GLOBAL_VARIABLES])
        if self.use_bias:
            self.b = tf.get_variable('bias',
                                     shape=[self.n_out, ],
                                     initializer=tf.zeros_initializer([self.n_out]),
                                     trainable=True,
                                     collections=[tf.GraphKeys.BIASES, GraphKeys.GLOBAL_VARIABLES]
                                     )
            self.normalizer_fn=None

        else:
            self.b = None
            self.normalizer_fn = normalizer_fn

    def call(self, inputs):
        outputs = standard_ops.matmul(inputs, self.W)

        if self.use_bias:
            outputs = nn.bias_add(outputs, self.b)

        # Apply normalizer function / layer.
        if self.normalizer_fn is not None:
            outputs = self.normalizer_fn(outputs)

        if self.activation_fn is not None:
            outputs = self.activation_fn(outputs)  # pylint: disable=not-callable

        return outputs