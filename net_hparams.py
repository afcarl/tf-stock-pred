import tensorflow as tf
from collections import namedtuple

# Training Parameters
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.flags.DEFINE_integer("batch_size", 20, "Batch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 20, "Batch size during evaluation")
tf.flags.DEFINE_string("optimizer", "SGD", "Optimizer Name (Adam, Adagrad, etc)")
tf.flags.DEFINE_integer('input_size', 120, 'Size of the input')
tf.flags.DEFINE_string("input_dir", "./data",
                       "Directory containing input data files 'train.tfrecords' and 'validation.tfrecords'")

FLAGS = tf.flags.FLAGS

HParams = namedtuple(
    "HParams",
    [
        "batch_size",
        "eval_batch_size",
        "learning_rate",
        "optimizer",
        "h_layer_size",
        "input_size"
    ])

def create_hparams():
    return HParams(
        batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        optimizer=FLAGS.optimizer,
        learning_rate=FLAGS.learning_rate,
        h_layer_size=[50, 50, 50],
        input_size=FLAGS.input_size
    )