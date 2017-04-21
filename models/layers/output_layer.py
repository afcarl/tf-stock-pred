import tensorflow as tf



def losses(logits, target, mode, h_params):
    if "class" in h_params.e_type:
        predictions = tf.argmax(tf.nn.softmax(logits), axis=1)

        if mode == tf.contrib.learn.ModeKeys.INFER:
            return predictions, None

        elif mode == tf.contrib.learn.ModeKeys.TRAIN:
            t_accuracy = tf.contrib.metrics.streaming_accuracy(predictions, target)
            tf.summary.scalar('train_accuracy', tf.reduce_mean(t_accuracy))

        # Calculate the binary cross-entropy loss
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=target, name='entropy')
    elif "reg" in h_params.e_type:
        predictions = tf.squeeze(logits, axis=1)

        if mode == tf.contrib.learn.ModeKeys.INFER:
            return predictions, None

        elif mode == tf.contrib.learn.ModeKeys.TRAIN:
            t_accuracy = tf.contrib.metrics.streaming_mean_squared_error(predictions, target)
            tf.summary.scalar('train_MSE', tf.reduce_mean(t_accuracy))

        # Calculate the binary cross-entropy loss
        losses = tf.sqrt(tf.losses.mean_squared_error(predictions=predictions, labels=target))  # RMSE
    else:
        return ValueError("Experiment type not defined")

    return predictions, losses