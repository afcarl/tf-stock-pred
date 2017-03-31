import os
import time

import tensorflow as tf
import model_input
import model_helper as helper
from models.multi_layer import multilayer_perceptron
from utils.eval_metric import create_evaluation_metrics
import utils.net_hparams as model_hparams


FLAGS = tf.flags.FLAGS

TIMESTAMP = int(time.time())
MODEL_DIR = os.path.abspath(os.path.join("./runs", str(TIMESTAMP)))


COMPANY_NAME = 'apple'
TRAIN_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, COMPANY_NAME, "train.tfrecords"))
VALIDATION_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, COMPANY_NAME, "valid.tfrecords"))

tf.logging.set_verbosity(FLAGS.loglevel)


def main(unused_argv):
    hparams = model_hparams.create_hparams()

    model_fn = helper.create_model_fn(
        hparams,
        model_impl=multilayer_perceptron)

    estimator = tf.contrib.learn.Estimator(
        model_fn=model_fn,
        model_dir=MODEL_DIR,
        config=tf.contrib.learn.RunConfig(gpu_memory_fraction=0.5))

    input_fn_train = model_input.create_input_fn(
        mode=tf.contrib.learn.ModeKeys.TRAIN,
        input_files=[TRAIN_FILE],
        batch_size=hparams.batch_size,
        num_epochs=FLAGS.num_epochs)

    input_fn_eval = model_input.create_input_fn(
        mode=tf.contrib.learn.ModeKeys.EVAL,
        input_files=[VALIDATION_FILE],
        batch_size=hparams.eval_batch_size,
        num_epochs=1)

    eval_metrics = create_evaluation_metrics()

    eval_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        input_fn=input_fn_eval,
        every_n_steps=FLAGS.eval_every,
        metrics=eval_metrics)

    estimator.fit(input_fn=input_fn_train, steps=None, monitors=[eval_monitor])


if __name__ == "__main__":
    tf.app.run()
