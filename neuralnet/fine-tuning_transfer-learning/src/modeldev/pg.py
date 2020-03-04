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
from sklearn.model_selection import train_test_split

from modeldev import d_layer
from mylib import utils
from mylib import validation_utils

random_state = 50


hypeopt_vae_layer_dim_list_pattern = [[10],
                                      [7, 9, 11],
                                      [5]]
a = hypeopt_vae_layer_dim_list_pattern

b = list([len(v) for v in a])
print(b)
