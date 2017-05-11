import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from models.layers import output_layer
import utils.summarizer as s


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def leaky_relu(x, alpha=.5, max_value=None):
    '''ReLU.

    alpha: slope of negative section.
    '''
    return tf.maximum(alpha * x, x)

def h_cnn_rnn(h_params, mode, features_map, target):
    features = features_map['features']
    sequence_length = features_map['length']

    in_channel = 1
    features = tf.expand_dims(features, -1)         # add channel dim
    #apply conv_filtering
    layers_output = []
    for idx, dim in enumerate(h_params.h_layer_size[:-2]): # -2 because there is the ouput layer and the linear projection
        if idx == 0:
            input_layer = features
            in_channel = 1
        else:
            input_layer = layers_output[-1]
            in_channel = h_params.h_layer_size[idx - 1][1]

        with tf.variable_scope('cnn_{}'.format(idx)) as vs:
            W = tf.get_variable('kernel', shape=[1, dim[0], in_channel, dim[1]])
            b = tf.get_variable('bias', shape=[dim[1]])
            layers_output.append(tf.nn.tanh(conv2d(input_layer, W) + b))
            s.add_kernel_summary(W, vs.name)

    # Concatenate the different filtered time_series
    features = tf.unstack(layers_output[-1], axis=3)
    features = tf.concat(features, axis=2)

    # apply linera transformation
    for layer_idx, h_layer_dim in enumerate(h_params.h_layer_size[2:-1]):
        layers_output = []
        with tf.variable_scope('ml_{}'.format(layer_idx), reuse=True) as vs:
            # Iterate over the timestamp
            for t in range(0, h_params.sequence_length):
                layer_output = tf.contrib.layers.fully_connected(inputs=features[:, t, :],
                                                                 num_outputs=h_layer_dim,
                                                                 activation_fn=leaky_relu,
                                                                 weights_initializer=initializers.xavier_initializer(),
                                                                 # normalizer_fn=tf.contrib.layers.layer_norm,
                                                                 scope=vs)

                # if h_params.dropout is not None and mode == tf.contrib.learn.ModeKeys.TRAIN:
                #     layer_output = tf.nn.dropout(layer_output, keep_prob=1 - h_params.dropout)

                layers_output.append(tf.expand_dims(layer_output, 1))  # add again the timestemp dimention to allow concatenation
            # proved to be the same weights
            s.add_hidden_layers_summary(layers_output, vs.name, weight=tf.get_variable("weights"))
        features = tf.concat(layers_output, axis=1)     # concat the different time_stamp

    # TODO: try attention transfomration
    # TODO: try an highway networks


    with tf.variable_scope('rnn') as vs:
        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)

        # Define a lstm cell with tensorflow
        cell = tf.contrib.rnn.LSTMCell(h_params.h_layer_size[-1],
                                       forget_bias=1.0,
                                       activation=tf.nn.tanh)
        # cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=h_params.dropout)

        # Get lstm cell output
        outputs, states = tf.contrib.rnn.static_rnn(cell, tf.unstack(features, axis=1),
                                                    sequence_length=sequence_length,
                                                    dtype=tf.float32)

        s.add_hidden_layers_summary(tensors=outputs, name=vs.name + "_output")
        s.add_hidden_layers_summary(tensors=states, name=vs.name + "_state")

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