import os
import pandas as pd

keys = ['Open', 'High', 'Low', 'Close', 'Volume']


def run(file_name, path='../../data', is_adjusted=False):
    if is_adjusted:
        file_name += '_adjusted'
    else:
        file_name += '_unadjusted'

    full_path = os.path.join(path, file_name) + '.txt'

    data = {}

    with open(full_path, 'r') as f:
        for ln_number, line in enumerate(f):
            tokens = line.strip().split(',')
            time = pd.Timestamp(tokens[0] + " " + tokens[1])
            row = {}
            for token, key in zip(tokens[2:], keys):
                if is_adjusted:
                    key = 'Adj_' + key
                row[key] = float(token)
            data[time] = row

    pd_data = pd.DataFrame.from_dict(data, orient='index')
    pd_data.index.name = 'Date'
    output_path = os.path.join('../../data/stock', file_name) + '.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pd_data.to_csv(output_path)


def merge(file_name, path='../../data/stock'):
    un_adj_data = pd.read_csv(os.path.join(path, file_name + "_unadjusted.csv"), parse_dates=True, index_col=0)

    adj_data = pd.read_csv(os.path.join(path, file_name + "_adjusted.csv"), index_col=0, parse_dates=True)


    data = pd.concat([un_adj_data, adj_data], axis=1)
    data = data.dropna()
    data.to_csv(os.path.join(path, file_name + ".csv"))



if __name__ == '__main__':
    # run('IBM', is_adjusted=False)
    # run('IBM', is_adjusted=True)
    merge('IBM')