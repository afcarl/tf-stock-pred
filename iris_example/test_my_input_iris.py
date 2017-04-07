from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import sys
from model_helper import create_model_fn
import net_hparams
from models.multi_layer import multilayer_perceptron
import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"


def model_fn_1(features, targets, mode, params):
    with tf.variable_scope('layer_1') as vs:
        first_hidden_layer = tf.contrib.layers.relu(features, 10)

    with tf.variable_scope('layer_2') as vs:
    # Connect the second hidden layer to first hidden layer with relu
        second_hidden_layer = tf.contrib.layers.relu(first_hidden_layer, 20)

    with tf.variable_scope('layer_3') as vs:
        third_hidden_layer = tf.contrib.layers.relu(second_hidden_layer, 10)

    with tf.variable_scope('softmax') as vs:
        # Connect the output layer to second hidden layer (no activation fn)
        output_layer = tf.contrib.layers.linear(third_hidden_layer, 3)

        prediction = tf.argmax(tf.nn.softmax(output_layer), 1)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_layer, labels=targets, name='entropy'))


    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        t_accuracy = tf.contrib.metrics.streaming_accuracy(prediction, targets)
        tf.summary.scalar('train_accuracy', tf.reduce_mean(t_accuracy))

    eval_metric_ops = {
        "accuracy":tf.contrib.metrics.streaming_accuracy(prediction, targets )
    }

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=params['learning_rate'],
        optimizer="SGD")

    return model_fn_lib.ModelFnOps(
        mode=mode,
        predictions=prediction,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )

def main(unused_argv):


    model_params = net_hparams.create_hparams()
    model_fn = create_model_fn(
        model_params,
        model_impl=multilayer_perceptron)

    nn = tf.contrib.learn.Estimator(model_fn=model_fn,
                                    model_dir='./model_dir',
                                    config=tf.contrib.learn.RunConfig(gpu_memory_fraction=0.5,
                                                                      save_checkpoints_secs=5))

    # Load datasets.
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TRAINING,
                                                                       features_dtype=np.float64, target_dtype=np.int)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TEST,
                                                                   features_dtype=np.float64, target_dtype=np.int)

    # input_fn_train = data_set_helper.create_input_fn(
    #     mode=tf.contrib.learn.ModeKeys.TRAIN,
    #     input_files=['../data/iris/train.tfrecords'],
    #     batch_size=10,
    #     num_epochs=None)
    #
    # input_fn_eval = data_set_helper.create_input_fn(
    #     mode=tf.contrib.learn.ModeKeys.EVAL,
    #     input_files=['../data/iris/valid.tfrecords'],
    #     batch_size=5,
    #     num_epochs=1)



    validation_metrics = {"accuracy": tf.contrib.metrics.streaming_accuracy,
                          "precision": tf.contrib.metrics.streaming_precision,
                          "recall": tf.contrib.metrics.streaming_recall}

    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        test_set.data,
        test_set.target,
        # input_fn=input_fn_eval,
        every_n_steps=50,
        metrics=validation_metrics)

    # Fit
    nn.fit(steps=7000, monitors=[validation_monitor],
           x=training_set.data,
           y=training_set.target,
           # input_fn=input_fn_train
           )



    # Evaluate accuracy.
    ev = nn.evaluate(steps=1,
                     x=test_set.data,
                     y=test_set.target
                     # input_fn=input_fn_eval
                     )

    print('Accuracy: {0:f}'.format(ev['accuracy']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--train_data", type = str, default = "", help = "Path to the training data.")
    parser.add_argument(
        "--test_data", type = str, default = "", help = "Path to the test data.")
    parser.add_argument(
        "--predict_data",
        type = str,
        default = "",
        help = "Path to the prediction data.")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

