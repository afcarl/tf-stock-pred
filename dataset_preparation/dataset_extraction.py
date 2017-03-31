import tensorflow as tf
import functools
import numpy as np
import os
import pandas as pd

companies = ['apple', 'bank_of_america', 'cantel_medical_corp', 'capital_city_bank', 'goldman', 'google',
                 'ICU_medical', 'sunTrust_banks', 'wright_medical_group', 'yahoo']

header_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Split Ratio', 'Close-1', 'Open-1', 'Low-1', 'High-1', 'Volume-1', 'Split Ratio-1', 'Close-2', 'Open-2', 'Low-2', 'High-2', 'Volume-2', 'Split Ratio-2', 'Close-3', 'Open-3', 'Low-3', 'High-3', 'Volume-3', 'Split Ratio-3', 'Close-4', 'Open-4', 'Low-4', 'High-4', 'Volume-4', 'Split Ratio-4', 'Close-5', 'Open-5', 'Low-5', 'High-5', 'Volume-5', 'Split Ratio-5', 'Close-6', 'Open-6', 'Low-6', 'High-6', 'Volume-6', 'Split Ratio-6', 'Close-7', 'Open-7', 'Low-7', 'High-7', 'Volume-7', 'Split Ratio-7', 'Close-8', 'Open-8', 'Low-8', 'High-8', 'Volume-8', 'Split Ratio-8', 'Close-9', 'Open-9', 'Low-9', 'High-9', 'Volume-9', 'Split Ratio-9', 'Close-10', 'Open-10', 'Low-10', 'High-10', 'Volume-10', 'Split Ratio-10', 'Close-11', 'Open-11', 'Low-11', 'High-11', 'Volume-11', 'Split Ratio-11', 'Close-12', 'Open-12', 'Low-12', 'High-12', 'Volume-12', 'Split Ratio-12', 'Close-13', 'Open-13', 'Low-13', 'High-13', 'Volume-13', 'Split Ratio-13', 'Close-14', 'Open-14', 'Low-14', 'High-14', 'Volume-14', 'Split Ratio-14', 'Close-15', 'Open-15', 'Low-15', 'High-15', 'Volume-15', 'Split Ratio-15', 'Close-16', 'Open-16', 'Low-16', 'High-16', 'Volume-16', 'Split Ratio-16', 'Close-17', 'Open-17', 'Low-17', 'High-17', 'Volume-17', 'Split Ratio-17', 'Close-18', 'Open-18', 'Low-18', 'High-18', 'Volume-18', 'Split Ratio-18', 'Close-19', 'Open-19', 'Low-19', 'High-19', 'Volume-19', 'Split Ratio-19', 'Close+1']

tf.flags.DEFINE_string("input_dir", "../data/stock", "Directory containing input data files")
tf.flags.DEFINE_string("output_dir", "../data", "Directory containing output")
tf.flags.DEFINE_integer('batch_size', 30, 'size of the inputs batch')
tf.flags.DEFINE_integer('split_size', 450, 'size of each split')
tf.flags.DEFINE_float('train_cf', 4.5, 'coefficient of training size')
tf.flags.DEFINE_float('valid_cf', 1., 'coefficient of validation size')

FLAGS = tf.flags.FLAGS

def create_tfrecords_file(input, output_file_name, example_fn, path='../data'):
    """
    Creates a TFRecords file for the given input data and example transofmration function
    """
    full_path = os.path.join(path + '/' + output_file_name)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    writer = tf.python_io.TFRecordWriter(full_path)
    print("Creating TFRecords file at {}...".format(full_path))
    for feas, (label_time, label_value) in zip(input[0].itertuples(), input[1].iteritems()):
        assert feas[0] == label_time
        x = example_fn(list(feas[1:]), label_value)
        writer.write(x.SerializeToString())

    writer.close()
    print("Wrote to {}".format(full_path))



def create_example(features, label):
    """
    Creates a training example.
    Returnsthe a tensorflow.Example Protocol Buffer object.
    """
    example = tf.train.Example()
    example.features.feature["label"].int64_list.value.append(int(label))
    example.features.feature["features"].float_list.value.extend(features)
    return example
    # write the new example




def split_train_valid_test(data, split, train_cff, valid_cff, experiment_type='classification'):
    train_size = int(split * train_cff)
    valid_size = train_size + int(split*valid_cff)

    if experiment_type == 'classification':
        for (close_time, close_value), (pred_time, pred_value) in zip(data['Close'].iteritems(), data['Close+1'].iteritems()):
            assert close_time == pred_time
            if pred_value >= close_value:
                data['Close+1'].set_value(pred_time, 1.)
            else:
                data['Close+1'].set_value(pred_time, 0.)

    y = data['Close+1']
    X = data.drop('Close+1', axis=1)

    X_train = X.ix[:train_size]
    y_train = y.ix[:train_size]

    X_valid = X.ix[train_size:valid_size]
    y_valid = y.ix[train_size:valid_size]

    X_test = X.ix[valid_size:]
    y_test = y.ix[valid_size:]

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

def run(file_name, path = '../data/stock'):


    full_path = os.path.join(path, file_name) + '-fea.csv'
    data = pd.read_csv(full_path, header=None, parse_dates=True, index_col="Date", names=header_names, skiprows=1)

    train, valid, test = split_train_valid_test(data, FLAGS.split_size, FLAGS.train_cf, FLAGS.valid_cf)


    create_tfrecords_file(
        input=train,
        output_file_name="train.tfrecords",
        example_fn=create_example,
        path=os.path.join(FLAGS.output_dir, file_name))

    # Create test.tfrecords
    create_tfrecords_file(
        input=test,
        output_file_name="test.tfrecords",
        example_fn=create_example,
        path=os.path.join(FLAGS.output_dir, file_name)
    )

    # Create train.tfrecords
    create_tfrecords_file(
        input=valid,
        output_file_name="valid.tfrecords",
        example_fn=create_example,
        path=os.path.join(FLAGS.output_dir, file_name)
    )

if __name__ == "__main__":
    run('apple')