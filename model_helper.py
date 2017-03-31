import tensorflow as tf
from utils.eval_metric import create_evaluation_metrics
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

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
            tf.summary.scalar('train_loss', loss)

            return probs, loss, train_op

        if mode == tf.contrib.learn.ModeKeys.INFER:
            probs, loss = model_impl(
                hparams,
                mode,
                feature,
                None)
            return probs, 0.0, None

        if mode == tf.contrib.learn.ModeKeys.EVAL:
            probs, loss = model_impl(
                hparams,
                mode,
                feature,
                target)

            # Add summaries
            tf.summary.histogram("eval_correct_probs_hist", probs[0])
            tf.summary.scalar("eval_correct_probs_average", tf.reduce_mean(probs[0]))
            tf.summary.histogram("eval_incorrect_probs_hist", probs[1])
            tf.summary.scalar("eval_incorrect_probs_average", tf.reduce_mean(probs[1]))

            return probs, loss, None
    return model_fn
