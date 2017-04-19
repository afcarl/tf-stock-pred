__author__ = 'andompesta'
import os
import pandas as pd
import utils.extraction_functions as ef
import net_hparams
import numpy as np

COMPANY_NAME = 'IBM'

def compute_label(close_time_serie, experiment_type):
    '''
    compute the label
    :param close_time_serie: closing price time-series
    :param experiment_type: type of the experiment: classification or regression
    '''
    # Compute the labels
    if experiment_type == 'classification':
        labels = ef.compute_label(close_time_serie)
    else:
        labels = ef.compute_delay(close_time_serie, -1)
        labels = ef.compute_return(labels)
    return labels

def compute_moving_average(close_time_serie, long_term=100, medium_term=5, short_term=10):
    '''
    compute the moving average of the given time series for different time interval
    :param close_time_serie: closing price time-series
    :param long_term: days of the long term MA
    :param medium_term: days for the medium MA
    :param short_term: days for the short MA
    '''
    long_term *= 60*7
    medium_term *= 60*7
    short_term *= 60*7

    long_moving_average = ef.compute_moving_average(close_time_serie, long_term)
    medium_moving_average = ef.compute_moving_average(close_time_serie, medium_term)
    short_moving_average = ef.compute_moving_average(close_time_serie, short_term)
    # Compute the moving average indicator
    long_MACD = short_moving_average - long_moving_average
    short_MACD = short_moving_average - medium_moving_average
    long_PPO = long_MACD / long_moving_average
    short_PPO = short_MACD / medium_moving_average

    return long_moving_average, medium_moving_average, short_moving_average, long_MACD, short_MACD, long_PPO, short_PPO


def run(company_name, path='../data/stock'):
    companies = ['apple', 'bank_of_america', 'cantel_medical_corp', 'capital_city_bank', 'goldman', 'google',
                 'ICU_medical', 'sunTrust_banks', 'wright_medical_group', 'yahoo', 'IBM_short']


    header_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'Ex-Dividend', 'Split Ratio',
                    'Adj_Open', 'Adj_High','Adj_Low', 'Adj_Close', 'Adj_Volume']

    primary_key = 'Close'
    h_params = net_hparams.create_hparams()


    full_path = os.path.join(path, company_name) + '.csv'
    data = pd.read_csv(full_path, parse_dates=True, index_col=0)

    print("extract feature of %s" % company_name)


    labels = compute_label(data[primary_key], h_params.experiment_type)
    (long_MA, medium_MA, short_MA, long_MACD, short_MACD, long_PPO, short_PPO) = compute_moving_average(data[primary_key])
    accumulator_distributio = ef.accumulation_distribution_line(data)

    # Compute prince relative returns
    data = ef.compute_return(data)

    # Insert new features
    data.insert(len(data.keys()), 'MA_long', long_MA)
    data.insert(len(data.keys()), 'MA_short', medium_MA)
    data.insert(len(data.keys()), 'MA_medium', short_MA)
    data.insert(len(data.keys()), 'MACD_long', long_MACD)
    data.insert(len(data.keys()), 'MACD_short', short_MACD)
    data.insert(len(data.keys()), 'PPO_long', long_PPO)
    data.insert(len(data.keys()), 'PPO_short', short_PPO)
    data.insert(5, 'A/D', accumulator_distributio)
    data.insert(len(data.keys()), 'Label', labels)

    data = ef.truncate_timeseries(data, pd.Timestamp('2001-01-01'), pd.Timestamp('2017-01-01'))

    keys = list(data.keys())
    for i in range(1, 60):
        for key in keys:
            data.insert(len(data.keys()), key+'-{}'.format(i), ef.compute_delay(data[key], i))
            # data.insert(len(data.keys()), 'Close-{}'.format(i), ef.compute_delay(data['Close'], i))
            # data.insert(len(data.keys()), 'Open-{}'.format(i), ef.compute_delay(data['Open'], i))
            # data.insert(len(data.keys()), 'Low-{}'.format(i), ef.compute_delay(data['Low'], i))
            # data.insert(len(data.keys()), "High-{}".format(i), ef.compute_delay(data['High'], i))
            # data.insert(len(data.keys()), "Volume-{}".format(i), ef.compute_delay(data['Volume'], i))
            # data.insert(len(data.keys()), "Adj_Close-{}".format(i), ef.compute_delay(data['Adj_Close'], i))
            # data.insert(len(data.keys()), "Adj_Open-{}".format(i), ef.compute_delay(data['Adj_Open'], i))
            # data.insert(len(data.keys()), "Adj_Low-{}".format(i), ef.compute_delay(data['Adj_Low'], i))
            # data.insert(len(data.keys()), "Adj_High-{}".format(i), ef.compute_delay(data['Adj_High'], i))
            # data.insert(len(data.keys()), "Adj_Volume-{}".format(i), ef.compute_delay(data['Adj_Volume'], i))
            # data.insert(len(data.keys()), 'MA_long-{}', long_moving_average)
            # data.insert(len(data.keys()), 'MA_short-{}', short_moving_average)
            # data.insert(len(data.keys()), 'MA_medium-{}', medium_moving_average)
            # data.insert(len(data.keys()), 'MACD_long-{}', long_MACD)
            # data.insert(len(data.keys()), 'MACD_short-{}', short_MACD)
            # data.insert(len(data.keys()), 'PPO_long-{}', long_PPO)
            # data.insert(len(data.keys()), 'PPO_short', short_PPO)
            #
            # data.insert(len(data.keys()), 'Label-{}'.format(i), ef.compute_delay(data['Label'], i))

    # remove unused feature
    # data = ef.remove_unused_key(data, remove_keys=['Adj_Close', 'Adj_Volume', 'Adj_High', 'Adj_Low', 'Adj_Open', 'Ex-Dividend'])

    # truncate the data accordingly to the value specified
    start_date = pd.Timestamp(h_params.start_time)
    end_date = pd.Timestamp(h_params.end_time)
    data = ef.truncate_timeseries(data, start_date, end_date)
    ef.check_data(data)

    data = data.astype(np.float32)
    data.to_csv(os.path.join(path, company_name) + '-fea.csv')


if __name__ == '__main__':
    run(COMPANY_NAME)

