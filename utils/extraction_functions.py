import pandas as pd

def compute_delay(series, delay):
    '''
    shift the time series of a given delay
    :param series: a company time series
    :param delay: delay to apply a the time series
    :return: x[t-delay]
    '''
    return series.shift(delay)

def predict_value(series):
    '''
    extract the value to predict
    :param series: time series that we what to predict (CLOSE price timeseries)
    :return: a timeseries of the value to predict
    '''
    return series.shift(-1)


def remove_unused_key(data, remove_keys=['Ex-Dividend']):
    '''
    :param data: the entire feature set extracted
    :return: only the used feature for the prediction
    '''
    for key in remove_keys:
        data = data.drop(key, 1)
    return data

def truncate_timeseries(data, start_date, end_date):
    '''
    truncate a time series to have all he values
    :param data: pandas dataframe to truncate
    :param start_date: starting date to keep
    :param end_date: ending date of the time frame
    :return: return a truncated time series
    '''
    data = data.ix[start_date: end_date]
    return data

def check_data(data):
    '''
    check if the dataframe hase nan values
    :param data: Dataframe contining the value
    :return: 
    '''
    if data.isnull().any().any():
        print(data.isnull().any())
        raise Exception("Error in the input values")

def compute_return(data):
    '''
    compute the return difference
    :param data: 
    :return: 
    '''
    shifted_data = data.shift(1)
    data = (data - shifted_data)/shifted_data
    return data

def compute_label(close_time_series):
    '''
    compute the classification label
    :param data: 
    :return: 
    '''
    close_time_series_shift = close_time_series.shift(-1)
    labels = close_time_series_shift - close_time_series

    for time, value in labels.iteritems():
        if value >= 0:
            labels[time] = 1
        else:
            labels[time] = 0
    return labels

def compute_moving_average(close_time_series, days):
    return close_time_series.rolling(window=days).mean()
