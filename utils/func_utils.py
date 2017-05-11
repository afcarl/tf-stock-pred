import tensorflow as tf
from collections import namedtuple
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import csv
import os
import pandas as pd

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

def parametric_relu(_x):
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)

    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg


def sum_regularizer(regularizer_list, scope=None):
    """Returns a function that applies the sum of multiple regularizers.
    Args:
        regularizer_list: A list of regularizers to apply.
        scope: An optional scope name
    Returns:
        A function with signature `sum_reg(weights)` that applies the
        sum of all the input regularizers.
    """
    regularizer_list = [reg for reg in regularizer_list if reg is not None]
    if not regularizer_list:
        return None

    def sum_reg(weights):
        """Applies the sum of all the input regularizers."""
        regularizer_tensors = [reg(weights) for reg in regularizer_list]
        return math_ops.add_n(regularizer_tensors)
    return sum_reg

def apply_regularization(regularizer, weights_list=None):
    """Returns the summed penalty by applying `regularizer` to the `weights_list`.

    Adding a regularization penalty over the layer weights and embedding weights
    can help prevent overfitting the training data. Regularization over layer
    biases is less common/useful, but assuming proper data preprocessing/mean
    subtraction, it usually shouldn't hurt much either.

    Args:
        regularizer: A function that takes a single `Tensor` argument and returns
            a scalar `Tensor` output.
        weights_list: List of weights `Tensors` or `Variables` to apply
            `regularizer` over. Defaults to the `GraphKeys.WEIGHTS` collection if
            `None`.

    Returns:
        A scalar representing the overall regularization penalty.

    Raises:
        ValueError: If `regularizer` does not return a scalar output, or if we find
            no weights.
    """

    if not weights_list:
        weights_list = ops.get_collection(ops.GraphKeys.WEIGHTS)
    if not weights_list:
        raise ValueError('No weights to regularize.')
    with ops.name_scope('get_regularization_penalty', values=weights_list) as scope:
        penalties = [regularizer(w) for w in weights_list]

        for p in penalties:
            if p.get_shape().ndims != 0:
                raise ValueError('regularizer must return a scalar Tensor instead of a '
                                 'Tensor with rank %d.' % p.get_shape().ndims)

        summed_penalty = math_ops.add_n(penalties, name=scope)
        ops.add_to_collection(ops.GraphKeys.REGULARIZATION_LOSSES, summed_penalty)
        return summed_penalty


def export_to_csv(file_name, data_dic, path='./data'):
    """
    export the data dictionary to a csv file
    :param file_name: 
    :param data_dic: 
    :param path: 
    :return: 
    """
    data_frame = pd.DataFrame(data_dic)
    full_path = os.path.join(path, file_name) + '.csv'
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, 'w') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, data_dic.keys())
        w.writeheader()
        for row in data_dic.items():
            w.writerow(row)


