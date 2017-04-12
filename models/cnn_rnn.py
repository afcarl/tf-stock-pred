import tensorflow as tf
from tensorflow.python.ops import nn
from tensorflow.contrib.layers.python.layers import initializers

def _add_hidden_layer_summary(value, tag):
  tf.summary.scalar("%s_fraction_of_zero_values" % tag, nn.zero_fraction(value))
  tf.summary.histogram("%s_activation" % tag, value)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

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
    with tf.variable_scope('cnn_one_by_one') as vs:
        W = tf.get_variable('kernel', shape=[1, 1, in_channel, h_params.one_by_one_out_filters])
        b = tf.get_variable('bias', shape=[h_params.one_by_one_out_filters])
        filter_one_by_one = tf.nn.tanh(conv2d(features, W) + b)
        # visualization

        # scale weights to [0 1], type is still float
        x_min = tf.reduce_min(W)
        x_max = tf.reduce_max(W)
        W_0_to_1 = (W - x_min) / (x_max - x_min)
        # to tf.image_summary format [batch_size, height, width, channels]
        W_transposed = tf.transpose(W_0_to_1, [3, 0, 1, 2])

        tf.summary.image(vs.name + '_filters', W_transposed, max_outputs=3)

    with tf.variable_scope('cnn_one_by_all') as vs:
        W = tf.get_variable('kernel', shape=[1, h_params.input_size, in_channel, h_params.one_by_all_out_filters])
        b = tf.get_variable('bias', shape=[h_params.one_by_all_out_filters])
        filter_one_by_all = tf.nn.tanh(conv2d(features, W) + b)

        # scale weights to [0 1], type is still float
        x_min = tf.reduce_min(W)
        x_max = tf.reduce_max(W)
        W_0_to_1 = (W - x_min) / (x_max - x_min)
        # to tf.image_summary format [batch_size, height, width, channels]
        W_transposed = tf.transpose(W_0_to_1, [3, 0, 1, 2])

        tf.summary.image(vs.name + '_filters', W_transposed, max_outputs=3)

    # Concatenate the different filtered time_series
    features = tf.unstack(filter_one_by_all, axis=3)
    features.extend(tf.unstack(filter_one_by_one, axis=3))
    features = tf.concat(features, axis=2)

    # apply linera transformation
    for layer_idx, h_layer_dim in enumerate(h_params.h_layer_size[:-1]):
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
            _add_hidden_layer_summary(tf.get_variable("weights"), vs.name + "_weights")
            _add_hidden_layer_summary(layers_output[-1], vs.name + "_activation")
        features = tf.concat(layers_output, axis=1)

    # TODO: linear layer transformation
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

        _add_hidden_layer_summary(outputs[-1], vs.name + "_output")
        _add_hidden_layer_summary(states[-1], vs.name + "_state")

    with tf.variable_scope('logits') as vs:
        logits = tf.contrib.layers.fully_connected(inputs=outputs[-1],
                                                   num_outputs=h_params.num_class,
                                                   activation_fn=None,
                                                   scope=vs)
        _add_hidden_layer_summary(logits, vs.name)
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