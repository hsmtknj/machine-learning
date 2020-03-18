import sys
from os import path, pardir
current_dir = path.abspath(path.dirname(__file__))
parent_dir = path.abspath(path.join(current_dir, pardir))
parent_parent_dir = path.abspath(path.join(parent_dir, pardir))
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from mylib import utils
from mylib import preproc_utils

#* Set params *#
# general params
random_state = 50
test_size_ratio = 0.1
valid_size_ratio = 0.1


if __name__ == '__main__':

    # load data
    dirname = parent_parent_dir + '/data/dataset'
    df_X, df_y = utils.load_dataset_X_y(dirname, 'pd')

    # ======================================================================= #
    # Data Preprocesing
    # ======================================================================= #

    # data preprocessing for all data
    # NOTE: do nothing curretly
    df_X = preproc_utils.data_preproc_base(df_X)

    # split into train, test and valid data
    split_data = preproc_utils.train_test_valid_split(df_X,
                                                      df_y,
                                                      test_size=test_size_ratio,
                                                      valid_size=valid_size_ratio,
                                                      random_state=random_state)
    df_train_X, df_test_X, df_valid_X, df_train_y, df_test_y, df_valid_y = split_data

    # data preprocessing for split data
    split_data = preproc_utils.data_preproc_split(df_train_X, 
                                                  df_valid_X,
                                                  df_test_X)
    df_train_X, df_valid_X, df_test_X = split_data

    # data preprocessing for train data
    df_train_X, df_train_y = preproc_utils.data_preproc_train(df_train_X,
                                                              df_train_y)
