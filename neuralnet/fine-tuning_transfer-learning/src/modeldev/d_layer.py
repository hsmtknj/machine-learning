"""
Layer Class Definition
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

# random generator and seed
rng = np.random.RandomState(1000)
random_state = 50

# layers class definition
class Dense(object):
    """
    Dense layer class
    """

    def __init__(self, input_dim, output_dim, act_func, name_W, name_b):
        """
        :param input_dim  : int, input dimension
        :param output_dim : int, output dimension
        :param act_func   : function, actimation function
        :param name_W     : str, weight name
        :param name_b     : str, bias name
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act_func = act_func
        self.name_W = name_W
        self.name_b = name_b

        # HACK: Initial value range is 0.08. <- Is this OK?
        self.W = tf.Variable(rng.uniform(low=-0.08, high=0.08, size=(input_dim, output_dim)).astype('float32'), name=name_W)
        self.b = tf.Variable(rng.uniform(low=-0.08, high=0.08, size=(output_dim)).astype('float32'), name=name_b)

    def f_prop(self, x):
        """
        forward propagation
        :param  x: tensor, input tensor
        :param  z: tensor, output tensor
        """
        u = tf.matmul(x, self.W) + self.b
        z = self.act_func(u)
        return z


class Drop(object):
    """
    Drop layer class

    NOTE: This is not implemented. DO NOT USE this class.
    """

    def __init__(self, input_dim, output_dim, act_func, name_W, name_b):
        """
        :param input_dim  : int, input dimension
        :param output_dim : int, output dimension
        :param act_func   : function, actimation function
        :param name_W     : str, weight name
        :param name_b     : str, bias name
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act_func = act_func
        self.name_W = name_W
        self.name_b = name_b

        # HACK: Initial value range is 0.08. <- Is this OK?
        self.W = tf.Variable(rng.uniform(low=-0.08, high=0.08, size=(input_dim, output_dim)).astype('float32'), name=name_W)
        self.b = tf.Variable(rng.uniform(low=-0.08, high=0.08, size=(output_dim)).astype('float32'), name=name_b)

    def f_prop(self, x):
        """
        forward propagation
        :param  x: tensor, input tensor
        :param  z: tensor, output tensor
        """
        u = tf.matmul(x, self.W) + self.b
        z = self.act_func(u)
        return z