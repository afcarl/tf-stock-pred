import tensorflow as tf
import functools
import numpy as np
import os
import pandas as pd
import net_hparams
from sklearn.model_selection import train_test_split

# companies = ['apple', 'bank_of_america', 'cantel_medical_corp', 'capital_city_bank', 'goldman', 'google',
#                  'ICU_medical', 'sunTrust_banks', 'wright_medical_group', 'yahoo']

# KEYS=['Open', 'High', 'Low', 'Close', 'Volume', 'A/D', 'Adj_Open', 'Adj_High','Adj_Low', 'Adj_Close', 'Adj_Volume', 'MA_long', 'MA_short', 'MA_medium', 'MACD_long', 'MACD_short', 'PPO_long', 'PPO_short']
# KEYS=['DEXUSAL', 'MA_long', 'MA_short', 'MA_medium', 'MACD_long', 'MACD_short', 'PPO_long', 'PPO_short']
h_params = net_hparams.create_hparams()

INPUT_DIR = "../data/stock"
OUTPUT_DIR = "../data"
COMPANY_NAME = "apple"
EXAMPLE_FN_NAME = "create_example_sequencial"
OUTPUT_NAME_SUFFIX = 'seq'
# example_fn = create_example
# output_name_suffix = ''

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

def create_example(row,  keys):
    """
    Creates a training example.
    Returnsthe a tensorflow.Example Protocol Buffer object.
    """
    label = row['Label']

    for i in range(1, h_params.sequence_length):
        row = row.drop("Label-{}".format(i))
    row = row.drop("Label")


    features = np.array(row[keys])
    example = tf.train.Example()

    if "class" in h_params.e_type:
        example.features.feature['label'].int64_list.value.append(np.int64(label))
    elif "reg" in h_params.e_type:
        example.features.feature['label'].float_list.value.extend(np.float32(label))
    else:
        return ValueError("error in the experiment type")

    example.features.feature["features"].float_list.value.extend(features)
    return example

def create_example_sequencial(row, keys):
    """
    Creates a training example.
    Returnsthe a tensorflow.Example Protocol Buffer object.
    """
    features = np.array(row).reshape(h_params.sequence_length, len(keys)+1)[::-1]
    example = tf.train.Example()
    example.features.feature["length"].int64_list.value.append(features.shape[0])

    for idx, key in enumerate(keys):
        example.features.feature[key].float_list.value.extend(features[:,idx])

    # fl_value = example.feature_lists.feature_list["label"]
    # fl_value.feature.add().int64_list.value.extend(np.int64(features[:, -1]))

    if "class" in h_params.e_type:
        example.features.feature['label'].int64_list.value.extend(np.int64(features[:, -1]))
    elif "reg" in h_params.e_type:
        example.features.feature['label'].float_list.value.extend(np.float32(features[:, -1]))
    else:
        return ValueError("error in the experiment type")
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

def run(file_name, example_fn_name, output_name_suffix, in_path = '../data/stock', out_path='../data'):
    example_fn = eval(example_fn_name)

    full_path = os.path.join(in_path, file_name) + '-{}-fea.csv'.format(h_params.e_type)
    print("processing {}".format(full_path))
    data = pd.read_csv(full_path, parse_dates=True, index_col=0)

    # data = pd.read_csv(full_path, header=None, parse_dates=True, index_col="Date", names=KEYS, skiprows=1)

    train, valid, test = split_train_valid_test(data)

    create_tfrecords_file(
        input=train,
        output_file_name="train_{}_{}.tfrecords".format(output_name_suffix, h_params.e_type),
        example_fn=functools.partial(example_fn, keys=h_params.KEYS),
        path=os.path.join(out_path, file_name))

    # Create test.tfrecords
    create_tfrecords_file(
        input=test,
        output_file_name="test_{}_{}.tfrecords".format(output_name_suffix, h_params.e_type),
        example_fn=functools.partial(example_fn, keys=h_params.KEYS),
        path=os.path.join(out_path, file_name)
    )

    # Create train.tfrecords
    create_tfrecords_file(
        input=valid,
        output_file_name="valid_{}_{}.tfrecords".format(output_name_suffix, h_params.e_type),
        example_fn=functools.partial(example_fn, keys=h_params.KEYS),
        path=os.path.join(out_path, file_name)
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
    #     path=os.path.join(out_path, file_name))
    #
    # create_tfrecords_file(
    #     input=(X_test, y_test),
    #     output_file_name="valid.tfrecords",
    #     example_fn=create_example,
    #     path=os.path.join(out_path, file_name)
    # )

if __name__ == "__main__":
    for company_name in ['apple']:
        run(company_name, EXAMPLE_FN_NAME, OUTPUT_NAME_SUFFIX, in_path=INPUT_DIR, out_path=OUTPUT_DIR)
