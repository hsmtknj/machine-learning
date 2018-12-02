import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import os


W = tf.Variable(np.arange(12).reshape(3, 4))

def func(W):
    return W * 2

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

W = func(W)

print(sess.run(W))
