import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
import utils.summarizer as s
import models.layers.conv_layer as conv_layer
import models.layers.output_layer as output_layer
import utils.func_utils as fu


def dw_cnn_rnn(h_params, mode, features_map, target):
    features = features_map['features']
    sequence_length = features_map['length']
    batch_norm_data = fu.create_BNParams(apply=True,
                                         phase=fu.is_training(mode))

    channel_multiply = 3
    features = tf.expand_dims(features, -2)         # add channel with dim






    filtered = conv_layer.depthwise_gated_conv1d(features,
                                                 filter_size=1,
                                                 in_channel=h_params.input_size,
                                                 channel_multiply=channel_multiply,
                                                 name="deepwise_gated_cnn",
                                                 activation_fn=tf.nn.elu,
                                                 batch_norm=batch_norm_data
                                                 )


    # reshape the data to a normal form -> move the input channel to the feature space
    filtered = tf.reshape(filtered, shape=[tf.shape(filtered)[0],       # last batch is not full
                                           h_params.sequence_length,
                                           h_params.input_size,
                                           channel_multiply])

    filtered = conv_layer.conv1d(filtered,
                                 filter_size=1,
                                 in_channel=channel_multiply,
                                 out_channel=1,
                                 name="cnn_down_sample",
                                 activation_fn=tf.nn.elu,
                                 batch_norm=batch_norm_data)

    filtered = tf.squeeze(filtered, axis=-1)



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
        logits = tf.contrib.layers.fully_connected(inputs=outputs[-1],
                                                   num_outputs=h_params.num_class[h_params.e_type],
                                                   activation_fn=None,
                                                   scope=vs)
        s.add_hidden_layer_summary(logits, vs.name)

        predictions, losses = output_layer.losses(logits, target, mode=mode, h_params=h_params)
        if mode == tf.contrib.learn.ModeKeys.INFER:
            return predictions, None


    mean_loss = tf.reduce_mean(losses, name='mean_loss')
    return predictions, mean_loss