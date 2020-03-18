# coding: utf-8

"""
Utilities Module for Data Preprocessing 
"""

#* Import Libraries *#
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#* Set Params *#
CLIPPING_PERCENTILE = 97
FEATURE_SCALING_OPT = 'minmax'

def data_preproc_base(df_X):
    """
    preprocess data for all data

    :param  df_X     : DataFrame, explanatory variable data
    :return df_proc_X: DataFrame, preprocessed data
    """    
    #* Label Encoding *# 
    # category_mapping = {'M':0, 'F':1}
    # df_X['Gender'] = df_X['Gender'].map(category_mapping)

    #* feature selection *#
    # drop feature: "Gender"
    # df_X = df_X.drop('Gender', axis=1)

    #* Increase Feature *#
    # add feature: polynomial feature
    # df_X = add_polynomial_feature(df_X)

    df_proc_X = df_X
    return df_proc_X


def normalize_feature_for_target_col(in_df, str_norm_target_col, abs_max_num):
    """
    normalize designated column
    e.g.
        normalize col with max "60"
           [20, 30, 70, 65, -90]
        -> [0.333..., 0.5, 1.0, 1.0, -1.0]
    
    :param  in_df               : pandas.DataFrame, 
    :param  str_norm_target_col : string, 
    :param  abs_max_num         : float, absolute max number (normalize col with this number)
    :return out_df              : pandas.DataFrame
    """

    assert(abs_max_num > 0), (
        'Please set positive number in "abs_max_num".'
    )

    # find target exceeding abs_max_num and overwrite the num with abs_max_num
    df = in_df

    # positive
    cond = (df.loc[:, str_norm_target_col] >= abs_max_num)
    df.loc[cond, str_norm_target_col] = abs_max_num

    # negative
    cond = (df.loc[:, str_norm_target_col] <= -abs_max_num)
    df.loc[cond, str_norm_target_col] = -abs_max_num
    
    # normalization
    df.loc[:, str_norm_target_col] = df.loc[:, str_norm_target_col] / abs_max_num

    out_df = df
    return out_df


def data_preproc_split(df_train_X, df_valid_X, df_test_X):
    """
    preprocess data with values which are extracted from traing data

    :param  df_train_X     : DataFrame
    :param  df_valid_X     : DataFrame
    :param  df_test_X      : DataFrame
    :return df_proc_train_X: DataFrame
    :return df_proc_valid_X: DataFrame
    :return df_proc_test_X : DataFrame
    """

    #* Clipping *#
    df_train_X, df_valid_X, df_test_X = clip_data(df_train_X,
                                                  df_valid_X,
                                                  df_test_X,
                                                  CLIPPING_PERCENTILE)

    #* Feature Scaling *#
    df_train_X, df_valid_X, df_test_X = feature_scaling(df_train_X,
                                                        df_valid_X,
                                                        df_test_X,
                                                        FEATURE_SCALING_OPT)

    #* Data Imputation *#
    # df_X = df_x_train.fillna(df_X.median())

    df_proc_train_X = df_train_X
    df_proc_valid_X = df_valid_X
    df_proc_test_X = df_test_X

    return df_proc_train_X, df_proc_valid_X, df_proc_test_X


def clip_data(df_train_X, df_valid_X, df_test_X, percent):
    """
    clip data to exclude outlier

    :param  df_train_X  : DataFrame
    :param  df_valid_X  : DataFrame
    :param  df_test_X   : DataFrame
    :param  percnet     : int, percentile [0 - 100]
    :return clipped data: DataFrame
    """
    # convert DataFrame to numpy
    train_X = df_train_X.values
    valid_X = df_valid_X.values
    test_X = df_test_X.values

    # get clipping threshold with train data
    upperbound, lowerbound = np.percentile(train_X, [100-percent, percent], axis=0)

    # clip data
    train_X = np.clip(train_X, upperbound, lowerbound)
    valid_X = np.clip(valid_X, upperbound, lowerbound)
    test_X = np.clip(test_X, upperbound, lowerbound)

    # convert numpy to DataFrame
    df_train_X = pd.DataFrame(train_X, columns=df_train_X.columns)
    df_valid_X = pd.DataFrame(valid_X, columns=df_train_X.columns)
    df_test_X = pd.DataFrame(test_X, columns=df_train_X.columns)

    return df_train_X, df_valid_X, df_test_X


def feature_scaling(df_train_X, df_valid_X, df_test_X, opt):
    """
    feature scaling: (X - X_mean) / X_var

    :param  df_train_X: DataFrame
    :param  df_valid_X: DataFrame
    :param  df_test_X : DataFrame
    :param  opt       : str, option ('minmax' or 'std')
    """
    # convert DataFrame to numpy
    train_X = df_train_X.values
    valid_X = df_valid_X.values
    test_X = df_test_X.values

    # get mean, variable and range from train data
    train_mean = np.mean(train_X, axis=0)
    train_var = np.mean(train_X, axis=0)
    train_minmax = np.max(train_X, axis=0) - np.min(train_X, axis=0)

    # feature scaling
    if (opt == 'minmax'):
        div = train_minmax
    elif (opt == 'std'):
        div = train_var

    train_X = (train_X - train_mean) / div
    valid_X = (valid_X - train_mean) / div
    test_X = (test_X - train_mean) / div

    # convert numpy to DataFrame
    df_train_X = pd.DataFrame(train_X, columns=df_train_X.columns)
    df_valid_X = pd.DataFrame(valid_X, columns=df_train_X.columns)
    df_test_X = pd.DataFrame(test_X, columns=df_train_X.columns)

    return df_train_X, df_valid_X, df_test_X


def train_test_valid_split(data_X, data_y, test_size, valid_size, random_state):
    """
    split data into "train", "test" and "valid" data
    e.g. total_data = 10000, test_size = 0.1, valid_size = 0.1
            train_data = 8000
            test_data = 1000
            valid_data = 1000

    :param  data_X       : split target data X
    :param  data_y       : split target data y
    :param  test_size    : test size ratio (0.0 - 1.0)
    :param  valid_size   : valid size ratio (0.0 - 1.0)
    :param  random_state : random seed
    :return split data   : 
    """

    # split data into test data
    test_data_size = test_size
    train_X, test_X, train_y, test_y = train_test_split(data_X,
                                                        data_y,
                                                        test_size=test_data_size,
                                                        random_state=random_state)
    # split data into valid data
    valid_data_size = valid_size / (1 - test_size)
    train_X, valid_X, train_y, valid_y = train_test_split(train_X,
                                                           train_y,
                                                           test_size=valid_data_size,
                                                           random_state=random_state)

    return train_X, test_X, valid_X, train_y, test_y, valid_y


def data_preproc_train(df_X, df_y):
    """
    preprocess train data only

    :param  df_X      : DataFrame
    :param  df_y      : DataFrame
    :return df_proc_X : DataFrame
    :return df_proc_y : DataFrame
    """
    df_proc_X = df_X
    df_proc_y = df_y

    #* Oversampling and Undersampling *#
    # sm = SMOTE(sampling_strategy='not majority',
    #            k_neighbors=4,
    #            random_state=random_state)

    # enn = EditedNearestNeighbours(sampling_strategy='majority',
    #                               n_neighbors=20,
    #                               kind_sel='mode')

    # sme = SMOTEENN(smote=sm,
    #                enn=enn,
    #                random_state=random_state)

    # proc_X, proc_y = enn.fit_sample(df_proc_X, df_proc_y)
    # proc_X, proc_y = sm.fit_sample(df_proc_X, df_proc_y)

    # df_proc_X = pd.DataFrame(proc_X)
    # df_proc_y = pd.DataFrame(proc_y)

    return df_proc_X, df_proc_y