import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
import utils.func_utils as fu

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
        clip_gradients=2.0,                                # clip gradient to a max value
        optimizer=hparams.optimizer)                        # optimizer used
    return train_op



def create_model_fn(hparams, model_impl):
    '''
    Function used to create the model according different implementations and usage mode
    :param hparams: hiper-parameters used to configure the model
    :param model_impl: implementation of the model used, have to use the same interface to inject a different model
    :return: probabilities of the predicted class, value of the loss function, operation to execute the training
    '''
    def model_fn(features_map, targets, mode):

        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            predictions, loss = model_impl(
                hparams,
                mode,
                features_map,
                targets)

            if hparams.l1_reg > 0. or hparams.l2_reg > 0.:
                # apply regularization
                with tf.variable_scope("reg") as vs:
                    all_regularize = []
                    if hparams.l1_reg > 0.:
                        all_regularize.append(tf.contrib.layers.l1_regularizer(hparams.l1_reg))
                    if hparams.l2_reg > 0.:
                        all_regularize.append(tf.contrib.layers.l2_regularizer(hparams.l2_reg))

                    regularizer = fu.sum_regularizer(all_regularize, scope=vs)
                    regularization_penalty = tf.contrib.layers.apply_regularization(regularizer)

                    loss += regularization_penalty

            train_op = create_train_op(loss, hparams)

            return model_fn_lib.ModelFnOps(
                mode=mode,
                predictions={'predictions':predictions,
                            'features':features_map['features'],
                            'targets':targets
                            },
                loss=loss,
                train_op=train_op
            )

        if mode == tf.contrib.learn.ModeKeys.INFER:
            predictions, loss = model_impl(
                hparams,
                mode,
                features_map,
                None)

            return model_fn_lib.ModelFnOps(mode=mode,
                                           predictions={'predictions':predictions,
                                                        'features':features_map['features'],
                                                        'targets':features_map['targets']
                                                        }
                                           )

        if mode == tf.contrib.learn.ModeKeys.EVAL:
            predictions, loss = model_impl(
                hparams,
                mode,
                features_map,
                targets)

            return model_fn_lib.ModelFnOps(mode=mode,
                                           predictions={'predictions':predictions,
                                                        'features':features_map['features'],
                                                        'targets':targets
                                                        },
                                           loss=loss
                                           )
    return model_fn
