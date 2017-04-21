import tensorflow as tf
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
from tensorflow.python.ops import math_ops, metrics_impl

def _streaming_mean_absolute_persentace_error(predictions, labels):
    absolute_errors = math_ops.abs((predictions - labels)/labels)
    mean_t, update_op = metrics_impl.mean(absolute_errors, None, None, None, 'mean_absolute_persentace_error')
    return tf.multiply(mean_t, 100.), update_op

def create_evaluation_metrics(e_type):
    """
    Create the appropriate eval metric according to the experiment type
    :param e_type: type of the experiment "classification" or "regression"
    :return: 
    """
    if "reg" in e_type:
        return _create_evaluation_metrics_regression()
    elif "clas" in e_type:
        return _create_evaluation_metrics_classify
    else:
        return ValueError("Wrong experiment type")


def _create_evaluation_metrics_classify():
    eval_metrics = {}
    eval_metrics['accuracy'] = MetricSpec(metric_fn=tf.contrib.metrics.streaming_accuracy,
                                          prediction_key="predictions")
    # eval_metrics['accuracy1'] = MetricSpec(metric_fn=evaluate_accuracy)

    eval_metrics['recall'] = MetricSpec(metric_fn=tf.contrib.metrics.streaming_recall,
                                        prediction_key="predictions")
    eval_metrics['precision'] = MetricSpec(metric_fn=tf.contrib.metrics.streaming_precision,
                                           prediction_key="predictions")
    return eval_metrics



def _create_evaluation_metrics_regression():
    eval_metrics = {}
    eval_metrics['MAE'] = MetricSpec(metric_fn=tf.contrib.metrics.streaming_mean_absolute_error,
                                     prediction_key="predictions")
    eval_metrics['MSE'] = MetricSpec(metric_fn=tf.contrib.metrics.streaming_mean_squared_error,
                                     prediction_key="predictions")
    return eval_metrics