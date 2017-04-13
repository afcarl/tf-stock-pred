import tensorflow as tf
import functools
import numpy as np
import os
import pandas as pd
import sklearn.datasets as sd
from sklearn.model_selection import train_test_split
# companies = ['apple', 'bank_of_america', 'cantel_medical_corp', 'capital_city_bank', 'goldman', 'google',
#                  'ICU_medical', 'sunTrust_banks', 'wright_medical_group', 'yahoo']

KEYS=['Open', 'High', 'Low', 'Close', 'Volume', 'A/D', 'Adj_Open', 'Adj_High','Adj_Low', 'Adj_Close', 'Adj_Volume', 'MA_long', 'MA_short', 'MA_medium', 'MACD_long', 'MACD_short', 'PPO_long', 'PPO_short']
# KEYS=['DEXUSAL', 'MA_long', 'MA_short', 'MA_medium', 'MACD_long', 'MACD_short', 'PPO_long', 'PPO_short']
TIME_STAMP=20

INPUT_DIR = "../data/stock"
OUTPUT_DIR = "../data"
COMPANY_NAME = "goldman"

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

def create_example(row,  keys=KEYS):
    """
    Creates a training example.
    Returnsthe a tensorflow.Example Protocol Buffer object.
    """
    label = int(row['Label'])

    for i in range(1,20):
        row = row.drop("Label-{}".format(i))
    row = row.drop("Label")


    features = np.array(row[keys])
    example = tf.train.Example()
    example.features.feature["label"].int64_list.value.append(label)
    example.features.feature["features"].float_list.value.extend(features)
    return example

def create_example_sequencial(row, keys=KEYS):
    """
    Creates a training example.
    Returnsthe a tensorflow.Example Protocol Buffer object.
    """
    features = np.array(row).reshape(TIME_STAMP, len(keys)+1)[::-1]
    example = tf.train.Example()
    example.features.feature["length"].int64_list.value.append(features.shape[0])

    for idx, key in enumerate(keys):
        example.features.feature[key].float_list.value.extend(features[:,idx])

    # fl_value = example.feature_lists.feature_list["label"]
    # fl_value.feature.add().int64_list.value.extend(np.int64(features[:, -1]))
    example.features.feature['label'].int64_list.value.extend(np.int64(features[:, -1]))
    return example
    # write the new example




def split_train_valid_test(data):
    # data_test = data.ix[pd.Timestamp("2015-01-01"):]
    # X_train = data.ix[:pd.Timestamp("2015-01-01")]
    # data_train, data_valid = train_test_split(X_train, test_size=0.3, random_state=42)

    data_test = data.ix[pd.Timestamp("2015-01-01"):]
    data_valid = data.ix[pd.Timestamp("2012-01-01"):pd.Timestamp("2015-01-01")]
    data_train = data.ix[:pd.Timestamp("2012-01-01")]
    return data_train, data_valid, data_test

def run(file_name, example_fn, output_name_suffix, path = '../data/stock'):

    full_path = os.path.join(path, file_name) + '-fea.csv'
    data = pd.read_csv(full_path, parse_dates=True, index_col=0)

    # data = pd.read_csv(full_path, header=None, parse_dates=True, index_col="Date", names=KEYS, skiprows=1)

    train, valid, test = split_train_valid_test(data)

    create_tfrecords_file(
        input=train,
        output_file_name="train"+output_name_suffix+".tfrecords",
        example_fn=functools.partial(example_fn, keys=KEYS),
        path=os.path.join(OUTPUT_DIR, file_name))

    # Create test.tfrecords
    create_tfrecords_file(
        input=test,
        output_file_name="test"+output_name_suffix+".tfrecords",
        example_fn=functools.partial(example_fn, keys=KEYS),
        path=os.path.join(OUTPUT_DIR, file_name)
    )

    # Create train.tfrecords
    create_tfrecords_file(
        input=valid,
        output_file_name="valid"+output_name_suffix+".tfrecords",
        example_fn=functools.partial(example_fn, keys=KEYS),
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

    example_fn = create_example_sequencial
    output_name_suffix = '_seq'
    # example_fn = create_example
    # output_name_suffix = ''

    run(COMPANY_NAME, example_fn, output_name_suffix, path=INPUT_DIR)