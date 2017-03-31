import tensorflow as tf



def multilayer_perceptron(hparams, mode, feature, target):
    logits = feature

    for layer_idx, h_layer_dim in enumerate(hparams.h_layer_size):
        if layer_idx == 0:
            in_dim = hparams.input_size
        else:
            in_dim = hparams.h_layer_size[layer_idx - 1]


        with tf.variable_scope('ml_{}'.format(layer_idx)) as vs:
            W = tf.get_variable('W', shape=[in_dim, h_layer_dim],
                                initializer=tf.truncated_normal_initializer())

            b = tf.get_variable('b', shape=[h_layer_dim],
                                initializer=tf.truncated_normal_initializer())

            activation = tf.add(tf.matmul(logits, W), b)
            l_output = tf.nn.relu(activation)
            logits = l_output
            tf.summary.histogram('ml_{}_W'.format(layer_idx), W)
            tf.summary.histogram('ml_{}_b'.format(layer_idx), b)
            tf.summary.histogram('ml_{}_output'.format(layer_idx), l_output)

    with tf.variable_scope('softmax_linear')as vs:
        W = tf.get_variable('W', shape=[hparams.h_layer_size[-1], 2],
                                initializer=tf.truncated_normal_initializer())

        b = tf.get_variable('b', shape=[2],
                            initializer=tf.truncated_normal_initializer())

        logits = tf.add(tf.matmul(logits, W), b)
        probs = tf.sigmoid(logits)

        tf.summary.histogram('softmax_linear_W', W)
        tf.summary.histogram('softmax_linear_b', b)
        tf.summary.histogram('softmax_linear_output', logits)

        if mode == tf.contrib.learn.ModeKeys.INFER:
            return probs, None

        # Calculate the binary cross-entropy loss
        # targets = tf.expand_dims(targets, 0)
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.to_float(target), name='entropy')

    mean_loss = tf.reduce_mean(losses, name='mean_loss')
    return probs, mean_loss