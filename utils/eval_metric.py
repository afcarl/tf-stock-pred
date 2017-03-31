import tensorflow as tf
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec


def create_evaluation_metrics():
    eval_metrics = {}
    eval_metrics['accuracy'] = MetricSpec(metric_fn=tf.contrib.metrics.streaming_accuracy)
    eval_metrics['recall'] = MetricSpec(metric_fn=tf.contrib.metrics.streaming_recall)
    eval_metrics['precision'] = MetricSpec(metric_fn=tf.contrib.metrics.streaming_precision)
    return eval_metrics