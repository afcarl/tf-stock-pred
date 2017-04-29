import tensorflow as tf

def leaky_relu(x, leakiness=.5, name=''):
    '''ReLU.

    alpha: slope of negative section.
    '''
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name=name+'_leaky_relu')


def is_training(mode):
    if mode == tf.contrib.learn.ModeKeys.INFER:
        return False
    else:
        return True