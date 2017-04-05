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
    def model_fn(feature, targets, mode):

        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            predictions, loss = model_impl(
                hparams,
                mode,
                feature,
                targets)
            train_op = create_train_op(loss, hparams)

            return model_fn_lib.ModelFnOps(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op
            )

        if mode == tf.contrib.learn.ModeKeys.INFER:
            predictions, loss = model_impl(
                hparams,
                mode,
                feature,
                None)

            return model_fn_lib.ModelFnOps(mode=mode,
                                           predictions={'predict':predictions,
                                                        'feature':feature
                                                        }
                                           )

        if mode == tf.contrib.learn.ModeKeys.EVAL:
            predictions, loss = model_impl(
                hparams,
                mode,
                feature,
                targets)

            eval_metric_ops = {
                # those are returned by the estimator
                "accuracy": tf.contrib.metrics.streaming_accuracy(predictions, targets)
            }

            return model_fn_lib.ModelFnOps(mode=mode,
                                           predictions=predictions,
                                           loss=loss,
                                           eval_metric_ops=eval_metric_ops)
    return model_fn
