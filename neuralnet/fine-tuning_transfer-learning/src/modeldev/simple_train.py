"""
Implement simple training
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
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
from modeldev import d_loss_func
from modeldev import d_calc_loss_func

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

    # set param
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
    sess = model.fit(sess, train_X, train_y, valid_X, valid_y)
    pred = model.predict(sess, test_X)

    error = np.mean((test_y - pred)**2)
    print('ERROR:{}'.format(error))

    print('END')
