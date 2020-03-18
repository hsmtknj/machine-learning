"""
Implement simple training
"""

#* Import Libraries *#
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

#* Register src directory path to PYTHONPATH *#
import sys
from os import path, pardir
current_dir = path.abspath(path.dirname(__file__))
parent_dir = path.abspath(path.join(current_dir, pardir))
parent_parent_dir = path.abspath(path.join(parent_dir, pardir))
sys.path.append(parent_dir)

#* Import my Libraries *#
from mylib import utils
from mylib import preproc_utils
from modeldev import d_layer
from modeldev import d_model
from modeldev import d_loss_func
from modeldev import d_calc_loss_func

#* Set params *#
# general params
random_state = 50
test_size_ratio = 0.1
valid_size_ratio = 0.1

# training params
vae_layer_dim_list = [13, 7, 5]
vae_layer_type_list = [d_layer.Dense, d_layer.Drop]
vae_act_func_list = [tf.nn.relu, tf.nn.relu6]
vae_loss_func = tf.losses.mean_squared_error
vae_optimizer = tf.train.AdamOptimizer
vae_epochs = 10
vae_kld_coef = 1.0
mlp_layer_dim_list = [3, 1]
mlp_layer_type_list = [d_layer.Dense, d_layer.Dense]
mlp_act_func_list = [tf.nn.relu, tf.nn.sigmoid]
mlp_loss_func = tf.losses.mean_squared_error
mlp_optimizer = tf.train.AdamOptimizer
mlp_epochs = 10


def simple_train_main():

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


    # ======================================================================= #
    # Training
    # ======================================================================= #

    # define model
    model = d_model.Model(vae_layer_dim_list,
                          vae_layer_type_list,
                          vae_act_func_list,
                          vae_loss_func,
                          vae_optimizer,
                          vae_epochs,
                          vae_kld_coef,
                          mlp_layer_dim_list,
                          mlp_layer_type_list,
                          mlp_act_func_list,
                          mlp_loss_func,
                          mlp_optimizer,
                          mlp_epochs)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # train
    sess = model.fit(sess, df_train_X.values, df_train_y.values,
                           df_valid_X.values, df_valid_y.values)
    pred = model.predict(sess, df_test_X.values)

    error = np.mean((df_test_y.values - pred)**2)
    print('ERROR:{}'.format(error))

    print('END')


if __name__ == '__main__':
    simple_train_main()