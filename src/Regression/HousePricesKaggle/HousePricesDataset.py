import csv
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


def one_hot_encode(data_matrix, cols_to_encode):
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False, drop='first')
    new_matrix = np.array(data_matrix[:, 0]).reshape(data_matrix.shape[0], 1)
    for col_idx in cols_to_encode:
        col = data_matrix[:, col_idx]
        integer_encoded = label_encoder.fit_transform(col)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        new_matrix = np.hstack((new_matrix, onehot_encoded))
    return new_matrix[:, 1:]


def quantitative_features(data_matrix, cols_to_quant):
    new_matrix = np.array(data_matrix[:, 0]).reshape(data_matrix.shape[0], 1)
    for col_idx in cols_to_quant:
        col = data_matrix[:, col_idx]
        numbers = np.where(col != "NA")[0]
        mean = col[numbers].astype('float').mean()
        col = np.where(col == "NA", mean, col).reshape((len(col), 1))
        new_matrix = np.hstack((new_matrix, col))
    return new_matrix[:, 1:]


class HousePricesDataset(Dataset):
    def __init__(self, train=True, random_seed=0):

        self.data = list()
        self.targets = list()
        data_matrix = []

        if train:
            with open('resources/train.csv', encoding="utf-8") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for line in csv_reader:
                    data_matrix.append(line)
            data_matrix = np.array(data_matrix)
            data_matrix = data_matrix[1:]
            to_one_hot_cols = [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31,
                               32, 33, 35, 39, 40, 41, 42, 53, 55, 57, 58, 60, 63, 64, 65, 72, 73, 74, 78, 79]
            to_quant_cols = [x for x in range(len(data_matrix[0])) if x not in to_one_hot_cols and x != 0]
            one_hot_features = one_hot_encode(data_matrix, to_one_hot_cols).astype(np.float)
            quant_features = quantitative_features(data_matrix, to_quant_cols).astype(np.float)
            features = np.hstack((one_hot_features, quant_features))
            self.data = torch.FloatTensor(features)
            self.targets = torch.FloatTensor(
                np.array(data_matrix[:, -1]).astype(np.float).reshape((len(data_matrix), 1)))

        else:
            with open('resources/test.csv', encoding="utf-8") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for line in csv_reader:
                    data_matrix.append(line)
            data_matrix = np.array(data_matrix)
            data_matrix = data_matrix[1:]
            to_one_hot_cols = [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31,
                               32, 33, 35, 39, 40, 41, 42, 53, 55, 57, 58, 60, 63, 64, 65, 72, 73, 74, 78, 79]
            to_quant_cols = [x for x in range(len(data_matrix[0])) if x not in to_one_hot_cols and x != 0]
            one_hot_features = one_hot_encode(data_matrix, to_one_hot_cols).astype(np.float)
            quant_features = quantitative_features(data_matrix, to_quant_cols).astype(np.float)
            features = np.hstack((one_hot_features, quant_features))
            self.data = torch.FloatTensor(features)
            self.targets = torch.FloatTensor(np.zeros((len(self.data), 1)))




    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


if __name__ == '__main__':
    tr_set = HousePricesDataset(train=True)
    print(len(tr_set))
