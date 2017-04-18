import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
import utils.summarizer as s
import models.layers.conv_layer as conv_layer
import models.layers.dense_layer as dense_layer


def leaky_relu(x, alpha=.5, max_value=None):
    '''ReLU.

    alpha: slope of negative section.
    '''
    return tf.maximum(alpha * x, x)


def is_training(mode):
    if mode == tf.contrib.learn.ModeKeys.INFER:
        return False
    else:
        return True

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
    filtered = tf.contrib.layers.batch_norm(filtered,
                                            center=True,
                                            scale=False,
                                            is_training=is_training(mode),
                                            scope='bn')
    # Concatenate the different filtered time_series
    # filtered = tf.unstack(filtered_one, axis=3)
    # filtered.extend(tf.unstack(filtered_all, axis=3))
    # filtered = tf.concat(filtered, axis=2)


    # dense_layer.dense_layer_over_time(filtered, h_params,
    #                                   activation_fn=leaky_relu)


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