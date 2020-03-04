"""
Hyperparama Optimization
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import csv
from itertools import product
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# register src directory path to PYTHONPATH
import sys
from os import path, pardir
current_dir = path.abspath(path.dirname(__file__))
parent_dir = path.abspath(path.join(current_dir, pardir))
parent_parent_dir = path.abspath(path.join(parent_dir, pardir))
sys.path.append(parent_dir)

from mylib import utils
from modeldev import d_layer
from modeldev import d_model

random_state = 50
test_size_ratio = 0.1
valid_size_ratio = 0.1

if __name__ == '__main__':
    # load data and split into train, test and valid data
    dirname = parent_parent_dir + '/data/dataset'
    data_X, data_y = utils.load_dataset_X_y(dirname)
    train_X, test_X, valid_X, train_y, test_y, valid_y = utils.train_test_valid_split(data_X,
                                                                                      data_y,
                                                                                      test_size=test_size_ratio,
                                                                                      valid_size=valid_size_ratio,
                                                                                      random_state=random_state)

    # =========================================================================
    # set hyperparameter
    # =========================================================================

    vae_epochs = int(10**1)
    mlp_epochs = int(10**1)

    hypeopt_vae_loss_func = ['tf.losses.mean_squared_error']
    hypeopt_vae_optimizer = ['tf.train.AdamOptimizer']
    hypeopt_mlp_loss_func = ['tf.losses.mean_squared_error']
    hypeopt_mlp_optimizer = ['tf.train.AdamOptimizer']

    # --- set model pattern ---
    hypeopt_vae_layer_dim_pattern_list = []
    hypeopt_vae_layer_type_pattern_list = []
    hypeopt_vae_act_func_pattern_list = []
    hypeopt_mlp_layer_dim_pattern_list = []
    hypeopt_mlp_layer_type_pattern_list = []
    hypeopt_mlp_act_func_pattern_list = []

    # set model
    hypeopt_vae_layer_dim_list_pattern = [[10],
                                          [7, 9, 11],
                                          [5]]

    hypeopt_vae_layer_type_list_pattern = [['d_layer.Dense'],
                                           ['d_layer.Dense']]

    hypeopt_vae_act_func_list_pattern = [['tf.nn.relu'],
                                         ['tf.nn.relu']]

    hypeopt_mlp_layer_dim_list_pattern = [[7, 9],
                                          [7, 9],
                                          [1]]

    hypeopt_mlp_layer_type_list_pattern = [['d_layer.Dense'],
                                           ['d_layer.Dense'],
                                           ['d_layer.Dense']]

    hypeopt_mlp_act_func_list_pattern = [['tf.nn.relu'],
                                         ['tf.nn.relu'],
                                         ['tf.nn.sigmoid']]

    hypeopt_vae_layer_dim_pattern_list.append(hypeopt_vae_layer_dim_list_pattern)
    hypeopt_vae_layer_type_pattern_list.append(hypeopt_vae_layer_type_list_pattern)
    hypeopt_vae_act_func_pattern_list.append(hypeopt_vae_act_func_list_pattern)
    hypeopt_mlp_layer_dim_pattern_list.append(hypeopt_mlp_layer_dim_list_pattern)
    hypeopt_mlp_layer_type_pattern_list.append(hypeopt_mlp_layer_type_list_pattern)
    hypeopt_mlp_act_func_pattern_list.append(hypeopt_mlp_act_func_list_pattern)


    # =========================================================================
    # find best combination of hyperparameter
    # =========================================================================

    # compute combination pattern number
    model_pattern_num = len(hypeopt_mlp_layer_dim_pattern_list)
    total_trial_num = 0
    for pattern in range(model_pattern_num):
        pattern_total_trial_num = 1

        vae_layer_dim_comb_num = np.prod([len(v) for v in hypeopt_vae_layer_dim_pattern_list[pattern]])
        vae_layer_type_comb_num = np.prod([len(v) for v in hypeopt_vae_layer_type_pattern_list[pattern]])
        vae_act_func_comb_num = np.prod([len(v) for v in hypeopt_vae_act_func_pattern_list[pattern]])
        mlp_layer_dim_comb_num = np.prod([len(v) for v in hypeopt_mlp_layer_dim_pattern_list[pattern]])
        mlp_layer_type_comb_num = np.prod([len(v) for v in hypeopt_mlp_layer_type_pattern_list[pattern]])
        mlp_act_func_comb_num = np.prod([len(v) for v in hypeopt_mlp_act_func_pattern_list[pattern]])

        pattern_total_trial_num *= vae_layer_dim_comb_num
        pattern_total_trial_num *= vae_layer_type_comb_num
        pattern_total_trial_num *= vae_act_func_comb_num
        pattern_total_trial_num *= mlp_layer_dim_comb_num
        pattern_total_trial_num *= mlp_layer_type_comb_num
        pattern_total_trial_num *= mlp_act_func_comb_num
        total_trial_num += pattern_total_trial_num

    print('TOTAL:{}'.format(total_trial_num))
