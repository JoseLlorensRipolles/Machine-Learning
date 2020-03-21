import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model.ridge
from sklearn.linear_model import Ridge


def fill_column(column):
    median = np.nanmedian(column)
    column[np.isnan(column)] = median
    return column


def normalise(column):
    return (column - np.mean(column)) / np.std(column)


def nan_count_per_feature(df):
    nan_count = dict()
    for feature in df.columns:
        nan_count[feature] = np.count_nonzero(df[feature].values == 'NA')/df.shape[0]
    nan_count = {k: v for k, v in sorted(nan_count.items(), key=lambda item: item[1]) if v > 0}
    plt.xticks(fontsize=10, rotation=70)
    plt.bar(nan_count.keys(), nan_count.values())
    plt.tight_layout()
    plt.show()


def categorical_features_histogram(df):
    features = ['Utilities']
    n = int(np.math.sqrt(len(features)))
    for idx, feature in enumerate(features):
        count = Counter(df[feature])
        count = {k: v for k, v in sorted(count.items(), key=lambda item: item[1])}
        plt.subplot(n, n, idx+1)
        plt.xticks(fontsize=10, rotation=70)
        plt.title(feature)
        plt.bar(count.keys(), count.values())
        plt.tight_layout()
    plt.show()


def quantitative_features_histogram(df):
    features = ['YearBuilt']
    n = int(np.math.sqrt(len(features)))
    for idx, feature in enumerate(features):
        values = df[feature][df[feature].notna()]
        plt.subplot(n, n, idx+1)
        plt.xticks(fontsize=10, rotation=70)
        plt.title(feature)
        sns.distplot(values)
        plt.tight_layout()
    plt.show()


def fill_lot_frontage(df):
    df_frontage = pd.get_dummies(df.drop('SalePrice', axis=1)).dropna()

    y = df_frontage.LotFrontage
    X = df_frontage.drop('LotFrontage', axis=1)

    model = Ridge()
    model.fit(X, y)

    X_ts = pd.get_dummies(df.drop('SalePrice', axis=1))[df['LotFrontage'].isna()].drop('LotFrontage', axis=1)
    df.loc[df['LotFrontage'].isna(), 'LotFrontage'] = model.predict(X_ts)

def fill_quantitative(df):
    df['BsmtFullBath'].fillna(0, inplace=True)
    df['BsmtHalfBath'].fillna(0, inplace=True)
    df['BsmtFinSF1'].fillna(0, inplace=True)
    df['BsmtFinSF2'].fillna(0, inplace=True)
    df['BsmtUnfSF'].fillna(0, inplace=True)
    df['TotalBsmtSF'].fillna(0, inplace=True)
    df.loc[df.GarageYrBlt.isnull(), 'GarageYrBlt'] = df.loc[df.GarageYrBlt.isnull(), 'YearBuilt']
    df['GarageArea'].fillna(0, inplace=True)
    df['GarageCars'].fillna(0, inplace=True)
    df['MasVnrArea'].fillna(0, inplace=True)


def fill_categorical(df):
    # False NaN
    df['Alley'].fillna('None', inplace=True)
    df['BsmtCond'].fillna('None', inplace=True)
    df['BsmtExposure'].fillna('None', inplace=True)
    df['BsmtFinType1'].fillna('None', inplace=True)
    df['BsmtFinType2'].fillna('None', inplace=True)
    df['BsmtQual'].fillna('None', inplace=True)
    df['Fence'].fillna('None', inplace=True)
    df['FireplaceQu'].fillna('None', inplace=True)
    df['GarageCond'].fillna('None', inplace=True)
    df['GarageFinish'].fillna('None', inplace=True)
    df['GarageQual'].fillna('None', inplace=True)
    df['GarageType'].fillna('None', inplace=True)
    df['MiscFeature'].fillna('None', inplace=True)
    df['PoolQC'].fillna('None', inplace=True)

    # Replace with mode
    df['SaleType'].fillna('WD', inplace=True)
    df['Utilities'].fillna('AllPub', inplace=True)
    df['Electrical'].fillna('SBrkr', inplace=True)
    df['Exterior1st'].fillna('VinylSd', inplace=True)
    df['Exterior2nd'].fillna('VinylSd', inplace=True)
    df['MSZoning'].fillna('RL', inplace=True)
    df['MasVnrType'].fillna('None', inplace=True)
    df['KitchenQual'].fillna('TA', inplace=True)
    df['Functional'].fillna('Typ', inplace=True)


def quantitative_features_vs_saleprice(df, quantitative_features):
    for feature in quantitative_features:
        x = df['YearBuilt']
        plt.xlabel('YrBuild')
        y = df[feature]
        plt.ylabel(feature)
        plt.scatter(x, y)
        plt.title(feature)
        plt.show()


def fix_remod(df):
    df.loc[df['YearRemodAdd'] == 1950, 'YearRemodAdd'] = df.loc[df['YearRemodAdd'] == 1950, 'YearBuilt']


if __name__ == '__main__':
    df_tr = pd.read_csv('./resources/train.csv')
    df_ts = pd.read_csv('./resources/test.csv')
    df = pd.concat([df_tr, df_ts])
    categorical_columns = np.array(['Alley', 'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                                 'BsmtQual', 'CentralAir', 'Condition1', 'Condition2', 'Electrical', 'ExterCond',
                                 'ExterQual', 'Exterior1st', 'Exterior2nd', 'Fence', 'FireplaceQu', 'Foundation',
                                 'Functional', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'Heating',
                                 'HeatingQC', 'HouseStyle', 'KitchenQual', 'LandContour', 'LandSlope', 'LotConfig',
                                 'LotShape', 'MSSubClass', 'MSZoning',  'MasVnrType', 'MiscFeature', 'Neighborhood',
                                 'PavedDrive', 'PoolQC', 'RoofMatl', 'RoofStyle', 'SaleCondition', 'SaleType', 'Street',
                                 'Utilities', 'MoSold'])

    quantitative_columns = np.array(['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BedroomAbvGr', 'BsmtFinSF1', 'BsmtFinSF2',
                         'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'EnclosedPorch', 'Fireplaces', 'FullBath',
                         'GarageArea', 'GarageCars', 'GarageYrBlt', 'GrLivArea', 'HalfBath', 'KitchenAbvGr',
                         'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'MiscVal',
                         'OpenPorchSF', 'OverallCond', 'OverallQual', 'PoolArea', 'ScreenPorch', 'TotRmsAbvGrd',
                         'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'YrSold'])

    fill_categorical(df)
    fill_quantitative(df)
    fill_lot_frontage(df)
    fix_remod(df)
    quantitative_features_vs_saleprice(df, ['YearRemodAdd'])