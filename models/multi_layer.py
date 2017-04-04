import tensorflow as tf



def multilayer_perceptron(hparams, mode, feature, target):
    layers_output = []
    for layer_idx, h_layer_dim in enumerate(hparams.h_layer_size):
        if layer_idx == 0:
            in_dim = hparams.input_size
            layer_input = feature
        else:
            in_dim = hparams.h_layer_size[layer_idx - 1]
            layer_input = layers_output[-1]

        with tf.variable_scope('ml_{}'.format(layer_idx)) as vs:
            W = tf.get_variable('W', shape=[in_dim, h_layer_dim],
                                initializer=tf.truncated_normal_initializer())

            b = tf.get_variable('b', shape=[h_layer_dim],
                                initializer=tf.truncated_normal_initializer())

            activation = tf.add(tf.matmul(layer_input, W), b)
            l_output = tf.nn.relu(activation)
            layers_output.append(l_output)
            tf.summary.histogram('ml_{}_W'.format(layer_idx), W)
            tf.summary.histogram('ml_{}_b'.format(layer_idx), b)
            tf.summary.histogram('ml_{}_output'.format(layer_idx), l_output)


    with tf.variable_scope('softmax_linear')as vs:
        W = tf.get_variable('W', shape=[hparams.h_layer_size[-1], hparams.num_class],
                                initializer=tf.truncated_normal_initializer())

        b = tf.get_variable('b', shape=[hparams.num_class],
                            initializer=tf.truncated_normal_initializer())

        logits = tf.add(tf.matmul(layers_output[-1], W), b)
        predictions = tf.argmax(tf.nn.softmax(logits), 1)

        tf.summary.histogram('softmax_linear_W', W)
        tf.summary.histogram('softmax_linear_b', b)
        tf.summary.histogram('softmax_linear_output', logits)

        if mode == tf.contrib.learn.ModeKeys.INFER:
            return predictions, None


        # Calculate the binary cross-entropy loss
        # target = tf.squeeze(target, 1)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=target, name='entropy')

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        t_accuracy = tf.contrib.metrics.streaming_accuracy(predictions, target)
        tf.summary.scalar('train_accuracy', tf.reduce_mean(t_accuracy))

    mean_loss = tf.reduce_mean(losses, name='mean_loss')
    return predictions, mean_loss