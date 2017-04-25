import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
import utils.summarizer as s
import models.layers.output_layer as output_layer
import models.layers.dense_layer as dense_layer

def parametric_relu(_x):
  alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5

  return pos + neg

def leaky_relu(x, alpha=.5, max_value=None):
    '''ReLU.

    alpha: slope of negative section.
    '''
    return tf.maximum(alpha * x, x)

def deep_rnn(h_params, mode, features_map, target):
    features = features_map['features']
    sequence_length = features_map['length']

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    # feature = tf.unstack(feature, h_params.sequence_length, 1)

    #apply unlinera transformation
    in_size = h_params.input_size
    filtered = features
    for layer_idx, h_layer_dim in enumerate(h_params.h_layer_size[:-1]):
        filtered = dense_layer.highway_dense_layer(filtered, in_size, h_layer_dim,
                                                   sequence_length=h_params.sequence_length,
                                                   scope_name='highway_{}'.format(layer_idx))
        in_size = h_layer_dim
        # layers_output = []
        # with tf.variable_scope('ml_{}'.format(layer_idx), reuse=True) as vs:
        #     # Iterate over the timestamp
        #     for t in range(0, h_params.sequence_length):
        #         layer_output = tf.contrib.layers.fully_connected(inputs=features[:, t, :],
        #                                                         num_outputs=h_layer_dim,
        #                                                         activation_fn=tf.tanh,
        #                                                         weights_initializer=initializers.xavier_initializer(),
        #                                                         # normalizer_fn=tf.contrib.layers.layer_norm,
        #                                                         scope=vs)
        #
        #         # if h_params.dropout is not None and mode == tf.contrib.learn.ModeKeys.TRAIN:
        #         #     layer_output = tf.nn.dropout(layer_output, keep_prob=1 - h_params.dropout)
        #         layers_output.append(tf.expand_dims(layer_output, 1))   # add again the timestemp dimention to allow concatenation
        #     # proved to be the same weights
        #     s.add_hidden_layer_summary(activation=layers_output[-1], weight=tf.get_variable("weights"), name=vs.name)
        # features = tf.concat(layers_output, axis=1)

    filtered += features
    with tf.variable_scope('rnn') as vs:
        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)

        # Define a lstm cell with tensorflow
        cell = tf.contrib.rnn.LSTMCell(h_params.h_layer_size[-1],
                                       forget_bias=1.0,
                                       activation=tf.nn.tanh)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=h_params.dropout)

        # Get lstm cell output
        outputs, states = tf.contrib.rnn.static_rnn(cell, tf.unstack(filtered, axis=1),
                                                    sequence_length=sequence_length,
                                                    dtype=tf.float32)

        s.add_hidden_layer_summary(activation=outputs[-1], name=vs.name + "_output")
        s.add_hidden_layers_summary(states, vs.name + "_state")

    with tf.variable_scope('logits') as vs:
        logits = tf.contrib.layers.fully_connected(inputs=outputs[-1],
                                                   num_outputs=h_params.num_class[h_params.e_type],
                                                   activation_fn=None,
                                                   scope=vs)
        s.add_hidden_layer_summary(logits, vs.name)

        predictions, losses = output_layer.losses(logits, target, mode=mode, h_params=h_params)

    mean_loss = tf.reduce_mean(losses, name='mean_loss')
    return predictions, mean_loss