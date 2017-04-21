import tensorflow as tf
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec



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
    eval_metrics['accuracy'] = MetricSpec(metric_fn=tf.contrib.metrics.streaming_accuracy)
    # eval_metrics['accuracy1'] = MetricSpec(metric_fn=evaluate_accuracy)

    eval_metrics['recall'] = MetricSpec(metric_fn=tf.contrib.metrics.streaming_recall)
    eval_metrics['precision'] = MetricSpec(metric_fn=tf.contrib.metrics.streaming_precision)
    return eval_metrics



def _create_evaluation_metrics_regression():
    eval_metrics = {}
    eval_metrics['MAE'] = MetricSpec(metric_fn=tf.contrib.metrics.streaming_mean_absolute_error)
    eval_metrics['MSE'] = MetricSpec(metric_fn=tf.contrib.metrics.streaming_mean_squared_error)
    # eval_metrics['MAPE'] = MetricSpec(metric_fn=tf.contrib.metrics.streaming_mean_relative_error)
    return eval_metrics