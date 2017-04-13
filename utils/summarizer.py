import tensorflow as tf
from tensorflow.python.ops import nn



def add_kernel_summary(kernel, tag):
    # visualization
    # scale weights to [0 1], type is still float
    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    # W_0_to_1 = (kernel - x_min) / (x_max - x_min)
    # to tf.image_summary format [batch_size, height, width, channels]
    W_transposed = tf.transpose(kernel, [3, 0, 1, 2])

    for row_idx, in_channel in enumerate(tf.unstack(W_transposed, axis=3)):
        for col_idx, filter in enumerate(tf.unstack(in_channel, axis=0)):
            tf.summary.scalar("{}_filters_in-channel-{}_out-channel-{}_norm".format(tag, row_idx, col_idx), tf.norm(filter))

        in_channel_norm = (in_channel - x_min) / (x_max - x_min)
        tf.summary.image(tag + '_filters_row_{}'.format(row_idx), tf.expand_dims(in_channel_norm, -1), max_outputs=col_idx+1)

def add_hidden_layers_summary(tensors, name, weight=None):
    '''
    create the summary of all the tensors passed
    :param tensors: list of tensor to summarize
    :param name: name to identify
    :param weight: weight to compute the norm
    '''
    if weight is not None:
        _norm_summary(weight=weight, name=name)
    return [add_hidden_layer_summary(activation=tensor, name=name+"_{}".format(i)) for i, tensor in enumerate(tensors)]

def _norm_summary(weight, name):
    tf.summary.scalar("%s_weight_norm" % name, tf.norm(weight))

def add_hidden_layer_summary(activation, name, weight=None):
    '''
    create the summary of the inout value
    usefull for debugging
    :param activation: tensor of the activation
    :param weight: tensor of the weight
    :param name: name of the tensor
    :return: zero_fraction, histogram, norm
    '''
    tf.summary.scalar("%s_fraction_of_zero_values" % name, nn.zero_fraction(activation))
    tf.summary.histogram("%s_activation" % name, activation)
    if weight is not None:
        _norm_summary(weight=weight, name=name)