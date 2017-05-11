import tensorflow as tf

def prediction_fn(logits, h_params):
    if "class" in h_params.e_type:
        return tf.argmax(tf.nn.softmax(logits), axis=1)
    elif "reg" in h_params.e_type:
        return tf.squeeze(logits, axis=1)
    else:
        return ValueError("Experiment type not defined")


def losses(logits, target, mode, h_params):
    predictions = prediction_fn(logits, h_params)

    if mode == tf.contrib.learn.ModeKeys.INFER:
        return predictions, None

    else:
        if "class" in h_params.e_type:
            if mode == tf.contrib.learn.ModeKeys.TRAIN:
                t_accuracy = tf.contrib.metrics.streaming_accuracy(predictions, target)
                tf.summary.scalar('train_accuracy', tf.reduce_mean(t_accuracy))

            # Calculate the binary cross-entropy loss
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=target, name='entropy')
        else:
            if mode == tf.contrib.learn.ModeKeys.TRAIN:
                t_performance = tf.contrib.metrics.streaming_mean_squared_error(predictions, target)
                tf.summary.scalar('train_MSE', tf.reduce_mean(t_performance))

            # Calculate the binary cross-entropy loss
            losses = tf.sqrt(tf.losses.mean_squared_error(predictions=predictions, labels=target))  # RMSE

        return predictions, losses