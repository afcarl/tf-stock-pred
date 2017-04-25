import os
import sys

import tensorflow as tf

import model_helper as model
import data_set_helper as data_set
import net_hparams
from models.multi_layer import mlp
from models.deep_rnn import deep_rnn
from models.simple_rnn import simple_rnn
from models.cnn_rnn import cnn_rnn
from models.hierarcical_cnn_rnn import h_cnn_rnn
from models.depthwise_cnn_rnn import dw_cnn_rnn

from utils.eval_metric import create_evaluation_metrics

tf.flags.DEFINE_integer("loglevel", 20, "Tensorflow log level")
tf.flags.DEFINE_string("model_dir", 'runs_1492760149', "Directory to load model checkpoints from")
tf.flags.DEFINE_string("input_dir", './data', "Evaluate after this many train steps")
FLAGS = tf.flags.FLAGS

if not FLAGS.model_dir:
    print("You must specify a model directory")
    sys.exit(1)

COMPANY_NAME = 'apple'
OUTPUT_NAME_SUFFIX = 'seq'

TEST_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, COMPANY_NAME, "test_{}_{}.tfrecords"))

tf.logging.set_verbosity(FLAGS.loglevel)




def main(unused_argv):
    hparams = net_hparams.create_hparams()

    model_impl = eval(hparams.model_type)

    model_fn = model.create_model_fn(
        hparams,
        model_impl=model_impl)

    estimator = tf.contrib.learn.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.model_dir,
        config=tf.contrib.learn.RunConfig(gpu_memory_fraction=0.6,
                                          save_checkpoints_secs=30))

    input_fn_test = data_set.create_input_fn(
        mode=tf.contrib.learn.ModeKeys.INFER,
        input_files=[TEST_FILE.format(OUTPUT_NAME_SUFFIX, hparams.e_type)],
        batch_size=hparams.eval_batch_size,
        num_epochs=1,
        h_params=hparams)


    ev = estimator.evaluate(input_fn=input_fn_test, steps=None)
    for idx_row, row in enumerate(ev):
        print("idx_row {}\t\tprediction {}\t\ttarget {}".format(idx_row, row['predictions'], row['targets']))



if __name__ == "__main__":
    tf.app.run()
