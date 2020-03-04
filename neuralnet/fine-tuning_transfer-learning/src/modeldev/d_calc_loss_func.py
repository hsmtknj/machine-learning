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

def calc_mean_squared_error(true, pred):
    """
    calculate loss with mean squared errro of tensorflow
    :param true:   np.ndarray
    :param pred:   np.ndarray
    :return error: int
    """

    input_true = true.rehsape(1, len(true))
    input_pred = pred.reshape(1, len(pred))

    true_y = tf.placeholder(dtype=np.dtype('float32'), shape=[None, len(true)])
    pred_y = tf.placeholder(dtype=np.dtype('float32'), shape=[None, len(pred)])
    loss = tf.losses.mean_squared_error(true_y, pred_y)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    error = sess.run(loss, feed_dict={true_y:input_true, pred_y:input_pred})

    return error
