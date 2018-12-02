"""
Stacked denoising Autoencoder
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.examples.tutorials.mnist import input_data

rng = np.random.RandomState(1000)
random_state = 50


# =============================================================================
# load data (mnist)
# =============================================================================

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
mnist_X, mnist_y = mnist.train.images, mnist.train.labels
train_X, valid_X, train_y, valid_y = train_test_split(mnist_X, mnist_y, test_size=0.1, random_state=random_state)

# cut down data size for small data set
train_X = train_X[0:100]
train_y = train_y[0:100]
valid_X = valid_X[0:100]
valid_y = valid_y[0:100]

# =============================================================================
# Define Layer Class
# =============================================================================

class Dense(object):
    """
    Dense class
    """
    def __init__(self, input_dim, output_dim, activation_func):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = tf.Variable(np.random.randn(input_dim, output_dim).astype('float32'))
        self.b = tf.Variable(np.random.randn(output_dim).astype('float32'))
        self.activation_func = activation_func
        self.ae = Autoencoder(input_dim, output_dim, activation_func, self.W )

    def f_prop(self, x):
        u = tf.matmul(x, self.W) + self.b
        z = self.activation_func(u)
        return z


class Autoencoder(object):
    """
    Autoencoder class
    """
    def __init__(self, input_dim, output_dim, activation_func, W):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = W
        self.b_encode = tf.Variable(np.zeros(output_dim).astype('float32'), name='b_encode')
        self.b_decode = tf.Variable(np.zeros(input_dim).astype('float32'), name='b_decode')
        self.activation_func = activation_func

    def encode(self, x):
        u = tf.matmul(x, self.W) + self.b_encode
        z = self.activation_func(u)
        return z

    def decode(self, x):
        u = tf.matmul(x, tf.transpose(self.W)) + self.b_decode
        z = self.activation_func(u)
        return z

    def compute_reconst_error(self, x, noise):
        # apply noise
        tilde = x * noise

        # compute reconstruction
        z = self.encode(x)
        reconst_x = self.decode(z)

        # reconstruction error
        error = -tf.reduce_mean(tf.reduce_sum(x * tf.log(tf.clip_by_value(reconst_x, 1e-10, 1)) + (1. - x) * tf.log(tf.clip_by_value(1 - reconst_x, 1e-10, 1)), axis=1))
        return error


layers = [Dense(784, 300, tf.nn.sigmoid),
          Dense(300, 10, tf.nn.softmax)]


# =============================================================================
# Pre-training
# =============================================================================

X = np.copy(train_X)

# set training parameters
epochs =  20
n_batches = 5
batch_size = train_X.shape[0] // n_batches
learning_ratio = 0.00001
corruption_level = np.float(0.3)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# train
for layer_num, layer in enumerate(layers):
    x = tf.placeholder(tf.float32, [None, layer.input_dim])
    noise = tf.placeholder(tf.float32, [None, layer.input_dim])

    cost_pre_train = layer.ae.compute_reconst_error(x, noise)
    pre_train = tf.train.GradientDescentOptimizer(learning_ratio).minimize(cost_pre_train)
    
    for epoch in range(epochs):
        # shuffle data
        X = shuffle(X, random_state=random_state)
        error_all = []
        for i in range(n_batches):
            start = batch_size * i
            end = start + batch_size
            noise_pre_training = rng.binomial(size=X[start:end, 0:layer.input_dim].shape, n=1, p=1-corruption_level)
            _, error = sess.run([pre_train, cost_pre_train], feed_dict={x: X[start:end, 0:layer.input_dim], noise: noise_pre_training})
            # layer.W = layer.ae.W
            error_all.append(error)
        print("[PRE-TRAINING] LAYER:{}, EPOCH:{}, ERROR:{}".format(layer_num+1, epoch+1, np.mean(error_all)))


# =============================================================================
# Fine-tuning
# =============================================================================

x_input = tf.placeholder(tf.float32, [None, 784])
noise_training = tf.placeholder(tf.float32, [None, 784])
y_fine_tuning = tf.placeholder(tf.float32, [None, 10])

# apply noise to input data
x_input = x_input * noise_training

# set training parameters
epochs =  100
n_batches = 5
batch_size = train_X.shape[0] // n_batches
learning_ratio = 0.00001
corruption_level = np.float(0.3)

# training model
u = x_input
for i, layer in enumerate(layers):
    u = layer.f_prop(u)
y = u

cost = -tf.reduce_mean(tf.reduce_sum(y_fine_tuning * tf.log( tf.clip_by_value(y, 1e-10, 1) ) + (1. - y_fine_tuning) * tf.log( tf.clip_by_value(1. - y, 1e-10, 1) ), axis=1))
train = tf.train.GradientDescentOptimizer(learning_ratio).minimize(cost)

# train
for epoch in range(epochs):
    # shuffle data
    train_X, train_y = shuffle(train_X, train_y, random_state=random_state)

    for i in range(n_batches):
        start = batch_size * i
        end = start + batch_size
        noise_training_ = rng.binomial(size=train_X[start:end].shape, n=1, p=1-corruption_level)
        sess.run(train, feed_dict={x_input: train_X[start:end], noise_training: noise_training_, y_fine_tuning: train_y[start:end]})

    error = sess.run(cost, feed_dict={x_input: valid_X, y_fine_tuning: valid_y})
    print("[FINE-TUNING] EPOCH:{}, ERROR:{}".format(epoch+1, error))
