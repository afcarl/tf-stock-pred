import tensorflow as tf
from tensorflow.python.ops import nn
from tensorflow.contrib.layers.python.layers import initializers
from models.layers import dense_layer

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

def leaky_relu(x, alpha=0.001):
    '''ReLU.

    alpha: slope of negative section.
    '''
    return tf.maximum(alpha * x, x)

def mlp(hparams, mode, features_map, target):
    layers_output = []
    features = features_map['features']

    for layer_idx, h_layer_dim in enumerate(hparams.h_layer_size):
        if layer_idx == 0:
            layer_input = features
        else:
            layer_input = layers_output[-1]

        with tf.variable_scope('ml_{}'.format(layer_idx)) as vs:

            layer_output = tf.contrib.layers.fully_connected(inputs=layer_input,
                                                             num_outputs=h_layer_dim,
                                                             activation_fn=leaky_relu,
                                                             weights_initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.5),
                                                             # weights_initializer=initializers.xavier_initializer(),
                                                             weights_regularizer=tf.contrib.layers.l2_regularizer(hparams.l2_reg),
                                                             # normalizer_fn=tf.contrib.layers.layer_norm,
                                                             scope=vs)

            if hparams.dropout is not None and mode == tf.contrib.learn.ModeKeys.TRAIN:
                layer_output = tf.nn.dropout(layer_output, keep_prob=1-hparams.dropout)

            _add_hidden_layer_summary(layer_output, vs.name)
            layers_output.append(layer_output)


    with tf.variable_scope('logits') as vs:
        logits = tf.contrib.layers.fully_connected(inputs=layers_output[-1],
                                                   num_outputs=hparams.num_class,
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