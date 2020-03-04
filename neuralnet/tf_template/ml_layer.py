"""
Machine learning layer class definition module
"""

import numpy as np
import tensorflow as tf

rng = np.random.RandomState(1000)

class Dense(object):
    """
    Dense layer class
    """

    def __init__(self):
        self.W = 0
        self.b = 0
        self.name_W = 'hoge'
        self.name_b = 'hoge'
        pass

    def f_prop(self):
        pass
