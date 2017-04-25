import functools
import numpy as np
import os
import pandas as pd
import net_hparams
import sklearn.datasets as sd
from sklearn.model_selection import train_test_split
# companies = ['apple', 'bank_of_america', 'cantel_medical_corp', 'capital_city_bank', 'goldman', 'google',
#                  'ICU_medical', 'sunTrust_banks', 'wright_medical_group', 'yahoo']

h_params = net_hparams.create_hparams()

INPUT_DIR = "../data/stock"
OUTPUT_DIR = "../data"
COMPANY_NAME = "apple"
EXAMPLE_FN_NAME = "create_example_sequencial"
OUTPUT_NAME_SUFFIX = 'seq'

def create_np_file(input, output_file_name, example_fn, path='../data'):
    """
    Creates a TFRecords file for the given input data and example transofmration function
    """
    full_path = os.path.join(path + '/' + output_file_name)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    data = example_fn(input)
    np.save(full_path, data)
    print("Wrote to {}".format(full_path))

def create_example(input,  keys):
    """
    Creates a training example.
    Returnsthe a tensorflow.Example Protocol Buffer object.
    """
    ret = np.zeros([input.shape()[0], len(keys)+1])
    ret[:, -1] = np.array(input['Label'])
    ret[:, :-2] = np.array(input[keys])
    return ret

def create_example_sequencial(input, keys):
    """
    Creates a training example.
    Returnsthe a tensorflow.Example Protocol Buffer object.
    """
    ret = np.zeros([input.shape[0], h_params.sequence_length, len(keys) + 1])
    for idx, (time, row) in enumerate(input.iterrows()):
        ret[idx, :, :] = np.array(row).reshape(h_params.sequence_length, len(keys)+1)[::-1]
    return ret



def split_train_valid_test(data):
    # data_test = data.ix[pd.Timestamp("2015-01-01"):]
    # X_train = data.ix[:pd.Timestamp("2015-01-01")]
    # data_train, data_valid = train_test_split(X_train, test_size=0.3, random_state=42)

    data_test = data.ix[pd.Timestamp("2015-01-01"):]
    data_valid = data.ix[pd.Timestamp("2012-01-01"):pd.Timestamp("2015-01-01")]
    data_train = data.ix[:pd.Timestamp("2012-01-01")]
    return data_train, data_valid, data_test

def run(file_name, example_fn_name, output_name_suffix, path = '../data/stock'):
    example_fn = eval(example_fn_name)

    full_path = os.path.join(path, file_name) + '-{}-fea.csv'.format(h_params.e_type)
    data = pd.read_csv(full_path, parse_dates=True, index_col=0)

    # data = pd.read_csv(full_path, header=None, parse_dates=True, index_col="Date", names=KEYS, skiprows=1)

    train, valid, test = split_train_valid_test(data)

    create_np_file(
        input=train,
        output_file_name="train_{}_{}".format(output_name_suffix, h_params.e_type),
        example_fn=functools.partial(example_fn, keys=h_params.KEYS),
        path=os.path.join(OUTPUT_DIR, file_name))

    # Create test.tfrecords
    create_np_file(
        input=test,
        output_file_name="test_{}_{}".format(output_name_suffix, h_params.e_type),
        example_fn=functools.partial(example_fn, keys=h_params.KEYS),
        path=os.path.join(OUTPUT_DIR, file_name)
    )

    # Create train.tfrecords
    create_np_file(
        input=valid,
        output_file_name="valid_{}_{}".format(output_name_suffix, h_params.e_type),
        example_fn=functools.partial(example_fn, keys=h_params.KEYS),
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


    run(COMPANY_NAME, EXAMPLE_FN_NAME, OUTPUT_NAME_SUFFIX, path=INPUT_DIR)