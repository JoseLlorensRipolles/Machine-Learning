import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge


def z_normalization(df, quantitative_columns):
    for feature in quantitative_columns:
        df[feature] = (df[feature] - df[feature].values.mean()) / df[feature].values.std()


def min_max_nomrmalization(df, quantitative_columns):
    for feature in quantitative_columns:
        df[feature] = (df[feature] - df[feature].values.min()) / (df[feature].values.max() - df[feature].values.min())


def correct_skew(df, quantitative_columns):
    skew_features = df[quantitative_columns].apply(lambda x: x.skew()).sort_values(ascending=False)

    high_skew = skew_features[skew_features > 0.5]
    skew_index = high_skew.index

    for i in skew_index:
        df[i] = np.log1p(df[i])


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

def fill_lot_frontage(df):
    df_frontage = pd.get_dummies(df.drop('SalePrice', axis=1)).dropna()

    y = df_frontage.LotFrontage
    X = df_frontage.drop('LotFrontage', axis=1)

    model = Ridge()
    model.fit(X, y)

    X_ts = pd.get_dummies(df.drop('SalePrice', axis=1))[df['LotFrontage'].isna()].drop('LotFrontage', axis=1)
    df.loc[df['LotFrontage'].isna(), 'LotFrontage'] = model.predict(X_ts)

def fix_remod(df):
    df.loc[df['YearRemodAdd'] == 1950, 'YearRemodAdd'] = df.loc[df['YearRemodAdd'] == 1950, 'YearBuilt']


def check_nans(df, quantitative_features):
    for feature in quantitative_features:
        a = df[feature]
        if np.any(np.isnan(a.values)):
            print(feature)


def dummy_quantitative(df):
    cols_to_create_dummy = ['2ndFlrSF', '3SsnPorch', 'BsmtFinSF1', 'BsmtFinSF2', 'EnclosedPorch', 'GarageArea',
                            'LowQualFinSF', 'MasVnrArea', 'MiscVal', 'OpenPorchSF', 'PoolArea', 'ScreenPorch',
                            'TotalBsmtSF', 'WoodDeckSF']
    for col in cols_to_create_dummy:
        df['Has_'+col] = (df[col]>0).astype(int)


def create_dataset():
    df_tr = pd.read_csv('./resources/train.csv', index_col=0)
    df_ts = pd.read_csv('./resources/test.csv', index_col=0)
    df = pd.concat([df_tr, df_ts])

    categorical_columns = np.array(['Alley', 'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                                    'BsmtQual', 'CentralAir', 'Condition1', 'Condition2', 'Electrical', 'ExterCond',
                                    'ExterQual', 'Exterior1st', 'Exterior2nd', 'Fence', 'FireplaceQu', 'Foundation',
                                    'Functional', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'Heating',
                                    'HeatingQC', 'HouseStyle', 'KitchenQual', 'LandContour', 'LandSlope', 'LotConfig',
                                    'LotShape', 'MSZoning', 'MasVnrType', 'MiscFeature', 'Neighborhood', 'PavedDrive',
                                    'PoolQC', 'RoofMatl', 'RoofStyle', 'SaleCondition', 'SaleType', 'Street',
                                    'Utilities'])

    quantitative_columns = np.array(['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BedroomAbvGr', 'BsmtFinSF1', 'BsmtFinSF2',
                                     'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'EnclosedPorch', 'Fireplaces',
                                     'FullBath',
                                     'GarageArea', 'GarageCars', 'GarageYrBlt', 'GrLivArea', 'HalfBath', 'KitchenAbvGr',
                                     'LotArea', 'LotFrontage', 'LowQualFinSF', 'MSSubClass', 'MasVnrArea', 'MiscVal',
                                     'MoSold',
                                     'OpenPorchSF', 'OverallCond', 'OverallQual', 'PoolArea', 'ScreenPorch',
                                     'TotRmsAbvGrd',
                                     'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'YrSold'])

    fill_categorical(df)
    fill_quantitative(df)
    fill_lot_frontage(df)
    fix_remod(df)
    dummy_quantitative(df)
    correct_skew(df, quantitative_columns)
    z_normalization(df, quantitative_columns)
    # min_max_nomrmalization(df, quantitative_columns)

    len_tr = 1460
    categorical_df = pd.get_dummies(df[categorical_columns], drop_first=True)
    quantitative_df = df[quantitative_columns]

    targets = df['SalePrice'].values[:len_tr].astype('float')
    df = pd.concat([categorical_df, quantitative_df.fillna(0)], axis=1)
    check_nans(df, quantitative_columns)
    matrix = df.values.astype('float')

    tr_data = torch.FloatTensor(matrix[:len_tr])
    targets = np.log1p(targets)
    mean = np.mean(targets)
    std = np.std(targets)
    print(mean, std)
    tr_targets = torch.FloatTensor(((targets - mean) / std).reshape((len_tr, 1)))
    ts_data = torch.FloatTensor(matrix[len_tr:])

    torch.save(tr_data, 'resources/tr_data.pt')
    torch.save(tr_targets, 'resources/tr_targets.pt')
    torch.save(ts_data, 'resources/ts_data.pt')


if __name__ == '__main__':
    create_dataset()
