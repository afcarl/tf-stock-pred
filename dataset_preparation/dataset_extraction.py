import tensorflow as tf
import functools
import numpy as np
import os
import pandas as pd
import sklearn.datasets as sd
from sklearn.model_selection import train_test_split
# companies = ['apple', 'bank_of_america', 'cantel_medical_corp', 'capital_city_bank', 'goldman', 'google',
#                  'ICU_medical', 'sunTrust_banks', 'wright_medical_group', 'yahoo']

companies = ['IBM']



INPUT_DIR = "../data/stock"
OUTPUT_DIR = "../data"

def create_tfrecords_file(input, output_file_name, example_fn, path='../data'):
    """
    Creates a TFRecords file for the given input data and example transofmration function
    """
    full_path = os.path.join(path + '/' + output_file_name)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    writer = tf.python_io.TFRecordWriter(full_path)
    print("Creating TFRecords file at {}...".format(full_path))
    for time, row in input.iterrows():
        x = example_fn(row)
        writer.write(x.SerializeToString())
    # for feature, target in zip(input[0], input[1]):
    #     x = example_fn(feature, target)
    #     writer.write(x.SerializeToString())

    writer.close()
    print("Wrote to {}".format(full_path))

def create_example(row, keys=['Open', 'High', 'Low', 'Close', 'Volume', 'Adj_Open', 'Adj_High','Adj_Low', 'Adj_Close', 'Adj_Volume', 'Close-1', 'Open-1', 'Low-1', 'High-1', 'Volume-1', 'Adj_Close-1', 'Adj_Open-1', 'Adj_Low-1', 'Adj_High-1', 'Adj_Volume-1', 'Close-2', 'Open-2', 'Low-2', 'High-2', 'Volume-2', 'Adj_Close-2', 'Adj_Open-2', 'Adj_Low-2', 'Adj_High-2', 'Adj_Volume-2', 'Close-3', 'Open-3', 'Low-3', 'High-3', 'Volume-3', 'Adj_Close-3', 'Adj_Open-3', 'Adj_Low-3', 'Adj_High-3', 'Adj_Volume-3', 'Close-4', 'Open-4', 'Low-4', 'High-4', 'Volume-4', 'Adj_Close-4', 'Adj_Open-4', 'Adj_Low-4', 'Adj_High-4', 'Adj_Volume-4', 'Close-5', 'Open-5', 'Low-5', 'High-5', 'Volume-5', 'Adj_Close-5', 'Adj_Open-5', 'Adj_Low-5', 'Adj_High-5', 'Adj_Volume-5', 'Close-6', 'Open-6', 'Low-6', 'High-6', 'Volume-6', 'Adj_Close-6', 'Adj_Open-6', 'Adj_Low-6', 'Adj_High-6', 'Adj_Volume-6', 'Close-7', 'Open-7', 'Low-7', 'High-7', 'Volume-7', 'Adj_Close-7', 'Adj_Open-7', 'Adj_Low-7', 'Adj_High-7', 'Adj_Volume-7', 'Close-8', 'Open-8', 'Low-8', 'High-8', 'Volume-8', 'Adj_Close-8', 'Adj_Open-8', 'Adj_Low-8', 'Adj_High-8', 'Adj_Volume-8', 'Close-9', 'Open-9', 'Low-9', 'High-9', 'Volume-9', 'Adj_Close-9', 'Adj_Open-9', 'Adj_Low-9', 'Adj_High-9', 'Adj_Volume-9', 'Close-10', 'Open-10', 'Low-10', 'High-10', 'Volume-10', 'Adj_Close-10', 'Adj_Open-10', 'Adj_Low-10', 'Adj_High-10', 'Adj_Volume-10', 'Close-11', 'Open-11', 'Low-11', 'High-11', 'Volume-11', 'Adj_Close-11', 'Adj_Open-11', 'Adj_Low-11', 'Adj_High-11', 'Adj_Volume-11', 'Close-12', 'Open-12', 'Low-12', 'High-12', 'Volume-12', 'Adj_Close-12', 'Adj_Open-12', 'Adj_Low-12', 'Adj_High-12', 'Adj_Volume-12', 'Close-13', 'Open-13', 'Low-13', 'High-13', 'Volume-13', 'Adj_Close-13', 'Adj_Open-13', 'Adj_Low-13', 'Adj_High-13', 'Adj_Volume-13', 'Close-14', 'Open-14', 'Low-14', 'High-14', 'Volume-14', 'Adj_Close-14', 'Adj_Open-14', 'Adj_Low-14', 'Adj_High-14', 'Adj_Volume-14', 'Close-15', 'Open-15', 'Low-15', 'High-15', 'Volume-15', 'Adj_Close-15', 'Adj_Open-15', 'Adj_Low-15', 'Adj_High-15', 'Adj_Volume-15', 'Close-16', 'Open-16', 'Low-16', 'High-16', 'Volume-16', 'Adj_Close-16', 'Adj_Open-16', 'Adj_Low-16', 'Adj_High-16', 'Adj_Volume-16', 'Close-17', 'Open-17', 'Low-17', 'High-17', 'Volume-17', 'Adj_Close-17', 'Adj_Open-17', 'Adj_Low-17', 'Adj_High-17', 'Adj_Volume-17', 'Close-18', 'Open-18', 'Low-18', 'High-18', 'Volume-18', 'Adj_Close-18', 'Adj_Open-18', 'Adj_Low-18', 'Adj_High-18', 'Adj_Volume-18', 'Close-19', 'Open-19', 'Low-19', 'High-19', 'Volume-19', 'Adj_Close-19', 'Adj_Open-19', 'Adj_Low-19', 'Adj_High-19', 'Adj_Volume-19']):
    """
    Creates a training example.
    Returnsthe a tensorflow.Example Protocol Buffer object.
    """

    features = np.array(row[keys])
    example = tf.train.Example()
    example.features.feature["label"].int64_list.value.append(int(row['Label']))
    example.features.feature["features"].float_list.value.extend(features)
    return example

def create_example_sequencial(row, keys=['Open', 'High', 'Low', 'Close', 'Volume', 'Adj_Open', 'Adj_High','Adj_Low', 'Adj_Close', 'Adj_Volume']):
    """
    Creates a training example.
    Returnsthe a tensorflow.Example Protocol Buffer object.
    """
    features = np.array(row).reshape(20, 11)
    example = tf.train.SequenceExample()
    example.context.feature["length"].int64_list.value.append(features.shape[0])

    for idx, key in enumerate(keys):
        fl_value = example.feature_lists.feature_list[key]
        fl_value.feature.add().float_list.value.extend(features[:,idx])

    fl_value = example.feature_lists.feature_list["Label"]
    fl_value.feature.add().int64_list.value.extend(np.int64(features[:, -1]))

    return example
    # write the new example




def split_train_valid_test(data):
    X_train, data_test = train_test_split(data, test_size=0.2, random_state=42)
    data_train, data_valid = train_test_split(X_train, test_size=0.3, random_state=42)
    return data_train, data_valid, data_test

def run(file_name, path = '../data/stock'):

    # example_fn = create_example_sequencial
    # output_name_suffix = '_seq'
    example_fn = create_example
    output_name_suffix = ''

    full_path = os.path.join(path, file_name) + '-fea.csv'
    if 'IBM' in file_name:
        data = pd.read_csv(full_path, parse_dates=True, index_col=0)
    else:
        data = pd.read_csv(full_path, header=None, parse_dates=True, index_col="Date", names=keys, skiprows=1)

    train, valid, test = split_train_valid_test(data)

    create_tfrecords_file(
        input=train,
        output_file_name="train"+output_name_suffix+".tfrecords",
        example_fn=example_fn,
        path=os.path.join(OUTPUT_DIR, file_name))

    # Create test.tfrecords
    create_tfrecords_file(
        input=test,
        output_file_name="test"+output_name_suffix+".tfrecords",
        example_fn=example_fn,
        path=os.path.join(OUTPUT_DIR, file_name)
    )

    # Create train.tfrecords
    create_tfrecords_file(
        input=valid,
        output_file_name="valid"+output_name_suffix+".tfrecords",
        example_fn=example_fn,
        path=os.path.join(OUTPUT_DIR, file_name)
    )


    # dataset_iris = sd.load_iris()
    # X_train, X_test, y_train, y_test = train_test_split(dataset_iris.data, dataset_iris.target, test_size=0.4,
    #                                                     random_state=0)
    #
    #
    # file_name = 'iris'
    # create_tfrecords_file(
    #     input=(X_train, y_train),
    #     output_file_name="train.tfrecords",
    #     example_fn=create_example,
    #     path=os.path.join(OUTPUT_DIR, file_name))
    #
    # create_tfrecords_file(
    #     input=(X_test, y_test),
    #     output_file_name="valid.tfrecords",
    #     example_fn=create_example,
    #     path=os.path.join(OUTPUT_DIR, file_name)
    # )

if __name__ == "__main__":
    run('IBM')