import tensorflow as tf
from collections import namedtuple


BNParams = namedtuple(
    "BNParams",
    [
        "apply",
        "phase",  # phase=Ture -> training
        "center",
        "scale"
    ])

def create_BNParams(apply=False, phase=True, center=True, scale=True):
    '''
    create the parameters for batch normalization
    :param apply: if we have to apply BN
    :param phase: True if training, False O.W.
    :param center: center the data
    :param scale: scale the data
    '''
    return BNParams(
        apply=apply,
        phase=phase,  # phase=Ture -> training
        center=center,
        scale=scale)

def leaky_relu(x, leakiness=.1, name=''):
    '''ReLU.

    alpha: slope of negative section.
    '''
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name=name+'_leaky_relu')


def is_training(mode):
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        return True
    else:
        return False