import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data


# ================================
# Define layers
# ================================

rng = np.random.RandomState(1000)
random_state = 50

# Convolution layer
class Conv:
    def __init__(self, filter_shape, function=lambda x: x, strides=[1,1,1,1], padding='VALID'):
        # Xavier Initialization
        fan_in = np.prod(filter_shape[:3])
        fan_out = np.prod(filter_shape[:2]) * filter_shape[3]
        self.W = tf.Variable(rng.uniform(
                        low=-np.sqrt(6/(fan_in + fan_out)),
                        high=np.sqrt(6/(fan_in + fan_out)),
                        size=filter_shape
                    ).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros((filter_shape[3]), dtype='float32'), name='b') # バイアスはフィルタごとなので, 出力フィルタ数と同じ次元数
        self.function = function
        self.strides = strides
        self.padding = padding

    def f_prop(self, x):
        u = tf.nn.conv2d(x, self.W, strides=self.strides, padding=self.padding) + self.b
        return self.function(u)

# Pooling layer
class Pooling:
    def __init__(self, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID'):
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    def f_prop(self, x):
        return tf.nn.max_pool(x, ksize=self.ksize, strides=self.strides, padding=self.padding)


# Flatten layer
class Flatten:
    def f_prop(self, x):
        return tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))

    
# Dense layer
class Dense:
    def __init__(self, in_dim, out_dim, function=lambda x: x):
        # Xavier Initialization
        self.W = tf.Variable(rng.uniform(
                        low=-np.sqrt(6/(in_dim + out_dim)),
                        high=np.sqrt(6/(in_dim + out_dim)),
                        size=(in_dim, out_dim)
                    ).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
        self.function = function

    def f_prop(self, x):
        return self.function(tf.matmul(x, self.W) + self.b)


if __name__ == '__main__':

    # ================================
    # Data Preprocessing
    # ================================

    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    mnist_X, mnist_y = mnist.train.images, mnist.train.labels
    mnist_X = mnist_X.reshape((mnist_X.shape[0], 28, 28, 1))

    train_X, valid_X, train_y, valid_y = train_test_split(mnist_X, mnist_y, test_size=0.1, random_state=50)
    
    
    # ================================
    # Define
    # ================================

    layers = [                             # (縦の次元数)x(横の次元数)x(チャネル数)
        Conv((5, 5, 1, 20), tf.nn.relu),   # 28x28x 1 -> 24x24x20
        Pooling((1, 2, 2, 1)),             # 24x24x20 -> 12x12x20
        Conv((5, 5, 20, 50), tf.nn.relu),  # 12x12x20 ->  8x 8x50
        Pooling((1, 2, 2, 1)),             #  8x 8x50 ->  4x 4x50
        Flatten(),
        Dense(4*4*50, 10, tf.nn.softmax)
    ]

    # define input
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    t = tf.placeholder(tf.float32, [None, 10])

    # define processing
    def f_props(layers, x):
        for layer in layers:
            x = layer.f_prop(x)
        return x

    # define output
    y = f_props(layers, x)

    # define cost computing and training (optimizing)
    # NOTE: prevent "nan" by computing tf.log(0)
    #       set 1e-10 in case a number is set to less than 1e-10
    cost = -tf.reduce_mean(tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), axis=1)) # tf.log(0)によるnanを防ぐ
    learning_ratio = 0.01
    train = tf.train.GradientDescentOptimizer(learning_ratio).minimize(cost)

    valid = tf.argmax(y, 1)


    # ================================
    # Run
    # ================================

    n_epochs = 5
    batch_size = 100
    n_batches = train_X.shape[0]//batch_size

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            train_X, train_y = shuffle(train_X, train_y, random_state=random_state)
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                sess.run(train, feed_dict={x: train_X[start:end], t: train_y[start:end]})
            pred_y, valid_cost = sess.run([valid, cost], feed_dict={x: valid_X, t: valid_y})
            print('EPOCH:: %i, Validation cost: %.3f, Validation F1: %.3f' % (epoch + 1, valid_cost, f1_score(np.argmax(valid_y, 1).astype('int32'), pred_y, average='macro')))
