import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


# reset graph
tf.reset_default_graph()

# =============================================================================
# Prepare Data
# =============================================================================

# training data
mu = 0
sigma = 10
data_num = 100
slope = 2
intercept = 1.5

data_x_ = np.random.normal(mu, sigma, data_num)
data_x = np.random.rand(100).astype(np.float32)
data_y = data_x * slope + intercept

# plt.scatter(data_x, data_y)
# plt.show()


# =============================================================================
# Define Part
# =============================================================================

# define model
W = tf.Variable(tf.random_uniform(shape=[1], minval=-1.0, maxval=1.0), dtype=tf.float32)
b = tf.Variable(tf.zeros(shape=[1]), dtype=tf.float32)
y = W * data_x + b

# define loss function
loss = tf.reduce_mean(tf.square(y - data_y))

# define optimizer
learning_ratio = 0.5
optimizer = tf.train.GradientDescentOptimizer(learning_ratio)
train = optimizer.minimize(loss)


# =============================================================================
# Run Part
# =============================================================================

epoch = 200

with tf.Session() as sess:
    # initialize tensor variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # training step
    for step in range(epoch):
        sess.run(train)

        if (step % 20 == 0):
            print(step, W.eval(), b.eval())
