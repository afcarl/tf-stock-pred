import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
import utils.eval_metric as eval_m

def create_train_op(loss, hparams):
    '''
    Function used to train the model
    :param loss: loss function to evaluate the error
    :param hparams: hiper-parameters used to configure the optimizer
    :return: training updates
    '''
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,                                          # loss function used
        global_step=tf.contrib.framework.get_global_step(), # number of batches seen so far
        learning_rate=hparams.learning_rate,                # learning rate
        clip_gradients=10.0,                                # clip gradient to a max value
        optimizer=hparams.optimizer)                        # optimizer used
    return train_op



def create_model_fn(hparams, model_impl):
    eval_metric_ops = eval_m.create_evaluation_metrics()
    '''
    Function used to create the model according different implementations and usage mode
    :param hparams: hiper-parameters used to configure the model
    :param model_impl: implementation of the model used, have to use the same interface to inject a different model
    :return: probabilities of the predicted class, value of the loss function, operation to execute the training
    '''
    def model_fn(feature, target, mode):

        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            probs, loss = model_impl(
                hparams,
                mode,
                feature,
                target)
            train_op = create_train_op(loss, hparams)

            # tf.summary.scalar('train_loss', loss)
            # tf.summary.scalar('train_accuracy',
            #                   tf.contrib.metrics.accuracy(tf.argmax(probs, axis=1), tf.argmax(target, axis=1)))
            # stram_accuracy = tf.contrib.metrics.streaming_accuracy(probs, target)
            # tf.summary.scalar('train_stream_accuracy', stram_accuracy[0])
            # tf.summary.scalar('train_stream_accuracy_up', stram_accuracy[1])
            ret = tf.contrib.learn.ModelFnOps(mode, probs, loss, train_op, eval_metric_ops)
            return ret

        if mode == tf.contrib.learn.ModeKeys.INFER:
            probs, loss = model_impl(
                hparams,
                mode,
                feature,
                None)
            ret = tf.contrib.learn.ModelFnOps(mode, probs, 0.0)
            return ret

        if mode == tf.contrib.learn.ModeKeys.EVAL:
            probs, loss = model_impl(
                hparams,
                mode,
                feature,
                target)

            ret = tf.contrib.learn.ModelFnOps(mode, probs, loss, None, eval_metric_ops)
            return ret
    return model_fn
