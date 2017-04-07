import tensorflow as tf
from tensorflow.python.ops import nn
from tensorflow.contrib.layers.python.layers import initializers

def _add_hidden_layer_summary(value, tag):
  tf.summary.scalar("%s_fraction_of_zero_values" % tag, nn.zero_fraction(value))
  tf.summary.histogram("%s_activation" % tag, value)

def parametric_relu(_x):
  alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5

  return pos + neg

def leaky_relu(x, alpha=5., max_value=None):
    '''ReLU.

    alpha: slope of negative section.
    '''
    return tf.maximum(alpha * x, x)

def rnn(h_params, mode, features_map, target):
    features = features_map['features']
    sequence_length = features_map['length']

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    # feature = tf.unstack(feature, h_params.sequence_length, 1)

    #apply unlinera transformation
    for layer_idx, h_layer_dim in enumerate(h_params.h_layer_size[:-1]):
        layers_output = []
        with tf.variable_scope('ml_{}'.format(layer_idx), reuse=True) as vs:
            # Iterate over the timestamp
            for t in range(0, h_params.sequence_length):
                layer_output = tf.contrib.layers.fully_connected(inputs=features[:, t, :],
                                                                num_outputs=h_layer_dim,
                                                                activation_fn=tf.nn.sigmoid,
                                                                weights_initializer=initializers.xavier_initializer(),
                                                                # normalizer_fn=tf.contrib.layers.layer_norm,
                                                                scope=vs)

                # if h_params.dropout is not None and mode == tf.contrib.learn.ModeKeys.TRAIN:
                #     layer_output = tf.nn.dropout(layer_output, keep_prob=1 - h_params.dropout)

                _add_hidden_layer_summary(layer_output, vs.name+"_{}".format(t))
                layers_output.append(tf.expand_dims(layer_output, 1))
        features = tf.concat(layers_output, axis=1)


    with tf.variable_scope('rnn') as vs:
        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)

        # Define a lstm cell with tensorflow
        cell = tf.contrib.rnn.BasicLSTMCell(h_params.h_layer_size[-1], forget_bias=1.0)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=h_params.dropout)

        # Get lstm cell output
        outputs, states = tf.contrib.rnn.static_rnn(cell, tf.unstack(features, axis=1),
                                                    sequence_length=sequence_length,
                                                    dtype=tf.float32)

        _add_hidden_layer_summary(outputs[-1], vs.name)
        # _add_hidden_layer_summary(states[-1], vs.name)

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