import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
import utils.summarizer as s
import models.layers.conv_layer as conv_layer


def leaky_relu(x, alpha=.5, max_value=None):
    '''ReLU.

    alpha: slope of negative section.
    '''
    return tf.maximum(alpha * x, x)

def cnn_rnn(h_params, mode, features_map, target):
    features = features_map['features']
    sequence_length = features_map['length']

    in_channel = 1
    features = tf.expand_dims(features, -1)         # add channel dim


    #apply conv_filtering
    # filtered_all = conv_layer.conv2d(features,
    #                            filter_size=[1, h_params.input_size],
    #                            in_channel=in_channel,
    #                            out_channel=h_params.one_by_all_out_filters,
    #                            name="cnn_all")

    # filtered_one = conv_layer.highway_conv2d(features,
    #                              filter_size=[1, 1],
    #                              in_channel=in_channel,
    #                              out_channel=h_params.one_by_one_out_filters,
    #                              name="highway_cnn_one")

    filtered_one = conv_layer.gated_conv2d(features,
                                           filter_size=[1, 1],
                                           in_channel=in_channel,
                                           out_channel=h_params.one_by_one_out_filters,
                                           name="gated_cnn")

    filtered = conv_layer.conv2d(filtered_one,
                                 filter_size=[1, 1],
                                 in_channel=h_params.one_by_one_out_filters,
                                 out_channel=1,
                                 name="cnn_down_sample",
                                 activation_fn=tf.nn.elu)

    # filtered = tf.add(filtered, features)

    filtered = tf.squeeze(filtered, axis=-1)
    # Concatenate the different filtered time_series
    # filtered = tf.unstack(filtered_one, axis=3)
    # filtered.extend(tf.unstack(filtered_all, axis=3))
    # filtered = tf.concat(filtered, axis=2)


    # apply linera transformation to reduce the dimension
    # layers_output = []
    # with tf.variable_scope('ml', reuse=True) as vs:
    #     # Iterate over the timestamp
    #     for t in range(0, h_params.sequence_length):
    #         layer_output = tf.contrib.layers.fully_connected(inputs=filtered[:, t, :],
    #                                                          num_outputs=h_params.h_layer_size[-2],
    #                                                          activation_fn=leaky_relu,
    #                                                          weights_initializer=initializers.xavier_initializer(),
    #                                                          # normalizer_fn=tf.contrib.layers.layer_norm,
    #                                                          scope=vs)
    #         # apply dropout
    #         # if h_params.dropout is not None and mode == tf.contrib.learn.ModeKeys.TRAIN:
    #         #     layer_output = tf.nn.dropout(layer_output, keep_prob=1 - h_params.dropout)
    #
    #         layers_output.append(tf.expand_dims(layer_output, 1))  # add again the timestemp dimention to allow concatenation
    #     # proved to be the same weights
    #     s.add_hidden_layer_summary(layers_output[-1], vs.name, weight=tf.get_variable("weights"))
    # filtered = tf.concat(layers_output, axis=1)


    with tf.variable_scope('rnn') as vs:
        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)

        # Define a lstm cell with tensorflow
        cell = tf.contrib.rnn.LSTMCell(h_params.h_layer_size[-1],
                                       forget_bias=1.0,
                                       activation=tf.nn.tanh)
        # cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=h_params.dropout)

        # Get lstm cell output
        outputs, states = tf.contrib.rnn.static_rnn(cell, tf.unstack(filtered, axis=1),
                                                    sequence_length=sequence_length,
                                                    dtype=tf.float32)

        s.add_hidden_layer_summary(activation=outputs[-1], name=vs.name + "_output")
        s.add_hidden_layers_summary(tensors=states, name=vs.name + "_state")

    with tf.variable_scope('logits') as vs:
        logits = tf.contrib.layers.fully_connected(inputs=outputs[-1],
                                                   num_outputs=h_params.num_class,
                                                   activation_fn=None,
                                                   scope=vs)
        s.add_hidden_layer_summary(logits, vs.name)
        predictions = tf.argmax(tf.nn.softmax(logits), 1)


        if mode == tf.contrib.learn.ModeKeys.INFER:
            return predictions, None

        elif mode == tf.contrib.learn.ModeKeys.TRAIN:
            t_accuracy = tf.contrib.metrics.streaming_accuracy(predictions, target)
            tf.summary.scalar('train_accuracy', tf.reduce_mean(t_accuracy))

        # Calculate the binary cross-entropy loss
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=target, name='entropy')

    mean_loss = tf.reduce_mean(losses, name='mean_loss')
    return predictions, mean_loss