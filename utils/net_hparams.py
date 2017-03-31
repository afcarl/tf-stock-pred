import tensorflow as tf
from collections import namedtuple

# Training Parameters
tf.flags.DEFINE_float("learning_rate", 0.1, "Learning rate")
tf.flags.DEFINE_integer("batch_size", 15, "Batch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 5, "Batch size during evaluation")
tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer Name (Adam, Adagrad, etc)")
tf.flags.DEFINE_integer('input_size', 120, 'Size of the input')
tf.flags.DEFINE_string("input_dir", "./data",
                       "Directory containing input data files 'train.tfrecords' and 'validation.tfrecords'")

tf.flags.DEFINE_integer("loglevel", 20, "Tensorflow log level")
tf.flags.DEFINE_integer("num_epochs", None, "Number of training Epochs. Defaults to indefinite.")
tf.flags.DEFINE_integer("eval_every", 200, "Evaluate after this many train steps")

FLAGS = tf.flags.FLAGS

FLAGS.h_layer_size = [20]

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
        h_layer_size=FLAGS.h_layer_size,
        input_size=FLAGS.input_size
    )