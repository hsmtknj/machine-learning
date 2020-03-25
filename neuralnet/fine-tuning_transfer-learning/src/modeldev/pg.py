# register src directory path to PYTHONPATH
import sys
from os import path, pardir
current_dir = path.abspath(path.dirname(__file__))
parent_dir = path.abspath(path.join(current_dir, pardir))
parent_parent_dir = path.abspath(path.join(parent_dir, pardir))
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from modeldev import d_layer
from mylib import utils
from mylib import validation_utils

random_state = 50


# hypeopt_vae_layer_dim_list_pattern = [[10],
#                                       [7, 9, 11],
#                                       [5]]
# a = hypeopt_vae_layer_dim_list_pattern

# b = list([len(v) for v in a])
# print(b)


# a = np.arange(30).reshape(10, 3)
# b = np.arange(30).reshape(10, 3) * 2

# print(np.mean(np.sum((a-b)**2, axis=1)) / 3)

# error = mean_squared_error(a, b)

# print(error)

for i in range(10):
    if (i % 3 == 0):
        print(i)