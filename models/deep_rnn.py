import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
import utils.summarizer as s
import models.layers.output_layer as output_layer
import models.layers.dense_layer as dense_layer
import utils.func_utils as fu
from utils.func_utils import leaky_relu


def deep_rnn(h_params, mode, features_map, target):
    features = features_map['features']
    sequence_length = features_map['length']


    #apply unlinera transformation
    in_size = h_params.input_size
    filtered = features
    batch_norm_data = fu.create_BNParams(apply=True,
                                         phase=fu.is_training(mode))

    for layer_idx, h_layer_dim in enumerate(h_params.h_layer_size[:-1]):
        # filtered = dense_layer.dense_layer_over_time(filtered, in_size, h_layer_dim,
        #                                              sequence_length=h_params.sequence_length,
        #                                              scope_name='dense_{}'.format(layer_idx),
        #                                              activation_fn=tf.nn.elu,
        #                                              batch_norm=batch_norm_data)

        filtered = dense_layer.gated_dense_layer_ot(filtered, in_size, h_layer_dim,
                                                     sequence_length=h_params.sequence_length,
                                                     scope_name='gated_dense_{}'.format(layer_idx),
                                                     activation_fn=leaky_relu,
                                                     batch_norm=batch_norm_data)
        in_size = h_layer_dim

    with tf.variable_scope('rnn') as vs:
        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)

        # Define a lstm cell with tensorflow
        cell = tf.contrib.rnn.GRUCell(h_params.h_layer_size[-1],
                                      activation=tf.nn.tanh)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=h_params.dropout)

        # Get lstm cell output
        outputs, states = tf.contrib.rnn.static_rnn(cell, tf.unstack(filtered, axis=1),
                                                    sequence_length=sequence_length,
                                                    dtype=tf.float32)

        s.add_hidden_layer_summary(activation=outputs[-1], name=vs.name + "_output")
        if isinstance(states, list) or isinstance(states, tuple):
            s.add_hidden_layer_summary(states.h, vs.name + "_state")
        else:
            s.add_hidden_layer_summary(states, vs.name + "_state")

    with tf.variable_scope('logits') as vs:
        logits = dense_layer.dense_layer(x=outputs[-1],
                                         in_size=h_params.h_layer_size[-1],
                                         out_size=h_params.num_class[h_params.e_type],
                                         scope=vs,
                                         activation_fn=None)

        s.add_hidden_layer_summary(logits, vs.name)

        predictions, losses = output_layer.losses(logits, target, mode=mode, h_params=h_params)

    mean_loss = tf.reduce_mean(losses, name='mean_loss')
    return predictions, mean_loss