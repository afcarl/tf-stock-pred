import tensorflow as tf
import utils.summarizer as s



def mlp(h_params, mode, features_map, target):
    layers_output = []
    features = features_map['features']
    # features = features_map
    for layer_idx, h_layer_dim in enumerate(h_params.h_layer_size):
        if layer_idx == 0:
            layer_input = features
        else:
            layer_input = layers_output[-1]

        with tf.variable_scope('ml_{}'.format(layer_idx), reuse=True) as vs:

            layer_output = tf.contrib.layers.fully_connected(inputs=layer_input,
                                                             num_outputs=h_layer_dim,
                                                             activation_fn=leaky_relu,
                                                             # weights_initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.5),
                                                             weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                             weights_regularizer=tf.contrib.layers.l2_regularizer(hparams.l2_reg),
                                                             # normalizer_fn=tf.contrib.layers.layer_norm,
                                                             scope=vs)

            if h_params.dropout is not None and mode == tf.contrib.learn.ModeKeys.TRAIN:
                layer_output = tf.nn.dropout(layer_output, keep_prob=1-h_params.dropout)

            s.add_hidden_layer_summary(activation=layer_output, weight=tf.get_variable("weights"), name=vs.name)
            layers_output.append(layer_output)


    with tf.variable_scope('logits') as vs:
        logits = tf.contrib.layers.fully_connected(inputs=layers_output[-1],
                                                   num_outputs=h_params.num_class[h_params.e_type],
                                                   activation_fn=None,
                                                   scope=vs)
        s.add_hidden_layer_summary(activation=logits, name=vs.name)
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