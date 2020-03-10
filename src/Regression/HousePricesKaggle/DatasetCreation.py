import csv
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def one_hot_encode(data_matrix, cols_to_encode):
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False, drop='first')
    new_matrix = np.array(data_matrix[:, 0]).reshape(data_matrix.shape[0], 1)
    for col_idx in cols_to_encode:
        col = data_matrix[:, col_idx]
        integer_encoded = label_encoder.fit_transform(col)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        col = onehot_encoder.fit_transform(integer_encoded).astype('float')
        mean = np.mean(col)
        std = np.std(col)
        col = (col - mean) / std
        new_matrix = np.hstack((new_matrix, col))
    return new_matrix[:, 1:]


def quantitative_features(data_matrix, cols_to_quant):
    new_matrix = np.array(data_matrix[:, 0]).reshape(data_matrix.shape[0], 1)
    for col_idx in cols_to_quant:
        col = data_matrix[:, col_idx]
        numbers = np.where(col != "NA")[0]
        mean = col[numbers].astype('float').mean()
        col = np.where(col == "NA", mean, col).reshape((len(col), 1)).astype('float')

        mean = np.mean(col)
        std = np.std(col)
        col = (col - mean)/std
        new_matrix = np.hstack((new_matrix, col))
    return new_matrix[:, 1:]


def get_skip_cols(data_matrix):
    skip_cols = list()
    for i in range(data_matrix.shape[1]):
        missing_values = np.sum(np.where(data_matrix[:, i] == 'NA', 1, 0))
        if missing_values/data_matrix.shape[0] > 0.4:
            skip_cols.append(i)
    print(skip_cols)
    return skip_cols


def create_dataset():

    data_matrix = []
    len_tr = 0
    len_ts = 0

    with open('resources/train.csv', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for line in csv_reader:
            data_matrix.append(line)
            len_tr += 1

    with open('resources/test.csv', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for line in csv_reader:
            line.append("0")
            data_matrix.append(line)
            len_ts += 1

    data_matrix = np.array(data_matrix)
    skip_cols = get_skip_cols(data_matrix)
    to_one_hot_cols = [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31,
                       32, 33, 35, 39, 40, 41, 42, 53, 55, 57, 58, 60, 63, 64, 65, 72, 73, 74, 78, 79]
    to_one_hot_cols = [x for x in to_one_hot_cols if x not in skip_cols]
    to_quant_cols = [x for x in range(len(data_matrix[0])) if x not in to_one_hot_cols and x != 0 and x not in skip_cols]
    one_hot_features = one_hot_encode(data_matrix, to_one_hot_cols).astype(np.float)
    quant_features = quantitative_features(data_matrix, to_quant_cols).astype(np.float)
    features = np.hstack((one_hot_features, quant_features))

    tr_data = torch.FloatTensor(features[0:len_tr, :-1])

    original_targets = np.log1p(np.array(data_matrix[0:len_tr, -1]).astype(np.float))
    mean = np.mean(original_targets)
    std = np.std(original_targets)
    print(mean, std)
    tr_targets = torch.FloatTensor(((original_targets - mean) / std).reshape((len_tr, 1)))

    ts_data = torch.FloatTensor(features[len_tr:, :-1])
    ts_targets = torch.FloatTensor(
        np.array(data_matrix[len_tr:, -1]).astype(np.float).reshape((len_ts, 1)))

    torch.save(tr_data, 'resources/tr_data.pt')
    torch.save(tr_targets, 'resources/tr_targets.pt')
    torch.save(ts_data, 'resources/ts_data.pt')
    torch.save(ts_targets, 'resources/ts_targets.pt')


if __name__ == '__main__':
    create_dataset()
