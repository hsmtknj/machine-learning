"""
Loss Function Definition
"""

# register src directory path to PYTHONPATH
import sys
from os import path, pardir
current_dir = path.abspath(path.dirname(__file__))
parent_dir = path.abspath(path.join(current_dir, pardir))
parent_parent_dir = path.abspath(path.join(parent_dir, pardir))
sys.path.append(parent_dir)

import numpy as np
import tensorflow as tf

def mean_squared_error(t, y):
    """
    sample loss function :mean squared error
    
    DO NOT use this function
    Use "tf.losses.mean_squared_error" instead

    :param  t    : tensor, true
    :param  y    : tensor, output from model
    :retrun loss : tensor, loss by loss function
    """
    loss = tf.reduce_mean(tf.pow(y-t, 2))
    return loss
