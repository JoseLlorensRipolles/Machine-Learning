import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from scipy.stats import norm, skew

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
    for col_idx in cols_to_quant:
        col = data_matrix[:, col_idx]
        numbers = np.where(col != "NA")[0]
        mean = col[numbers].astype('float').mean()
        col = np.where(col == "NA", mean, col).reshape((len(col), 1)).astype('float')
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


def fill_columns(df):
    for (col_name, col_data) in df.iteritems():
        idx = np.where(np.array(col_data != 'NA'))[0]
        values = col_data[idx]
        mean = values.mean()
        pass


if __name__ == '__main__':
    #df = pd.DataFrame(data=data_matrix[1:, 1:], index=data_matrix[1:,0], columns=data_matrix[0, 1:])
    df = pd.read_csv('./resources/train.csv')
    categorical_cols = np.array([1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31,
                        32, 33, 35, 39, 40, 41, 42, 53, 55, 57, 58, 60, 63, 64, 65, 72, 73, 74, 78, 79])

    # skewness and kurtosis
    print("Skewness: %f" % df['SalePrice'].skew())
    print("Kurtosis: %f" % df['SalePrice'].kurt())
    values = np.log1p(df['SalePrice'].to_numpy())
    mean = values.mean()
    std = values.std()
    x = np.linspace(0, values.max(), 100)
    y = stats.norm.pdf(x, mean, std)
    sns.distplot(values)
    plt.plot(x,y)
    plt.show()

    # quantititive_cols = [x for x in range(len(data_matrix[0])) if x not in categorical_cols]
    # quant_feat = quantitative_features(data_matrix, quantititive_cols)
    # to_drop = [df.columns[i] for i in categorical_cols]
    # # df = df.drop(labels=to_drop, axis='columns')
    # fill_columns(df)
    # corr = df.corr()
    # plt.matshow(corr)
    # plt.show()
