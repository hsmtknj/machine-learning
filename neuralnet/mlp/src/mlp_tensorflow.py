import os
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.examples.tutorials.mnist import input_data


rng = np.random.RandomState(1000)
random_state = 50

# reset graph
tf.reset_default_graph()


if __name__ == '__main__':
    # =========================================================================
    # Load Data
    # =========================================================================

    # load MNIST data
    #   train_X : (49500, 784), pixel size is between 0 and 1
    #   valid_X : (  500, 784), pixel size is between 0 and 1
    #   train_y : (49500,    ), NOT one-hot
    #   valid_y : (  500,    ), NOT one-hoe
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=False)
    mnist_X, mnist_y = mnist.train.images, mnist.train.labels
    train_X, valid_X, train_y, valid_y = train_test_split(mnist_X, mnist_y, test_size=0.1, random_state=random_state)    

    # try and error with small dataset
    train_X = train_X[:100]
    train_y = train_y[:100]
    valid_X = valid_X[:100]
    valid_y = valid_y[:100]


    # =========================================================================
    # Preproces Data
    # =========================================================================

    # convert one-hot
    train_y = np.eye(10)[train_y]
    valid_y = np.eye(10)[valid_y]
    

    # =========================================================================
    # Define Part of Tensorflow
    # =========================================================================

    # set hyper parameters
    input_dim = train_X.shape[1]
    intermediate_dim = 200
    class_num = train_y.shape[1]
    eta = 0.01

    # define input
    x = tf.placeholder(tf.float32, [None, input_dim])
    t = tf.placeholder(tf.float32, [None, class_num])

    # define variable
    W1 = tf.Variable(rng.uniform(low=-0.08, high=0.08, size=(input_dim, intermediate_dim)).astype('float32'), name='W1')
    b1 = tf.Variable(np.zeros(intermediate_dim).astype('float32'), name='b1')
    W2 = tf.Variable(rng.uniform(low=-0.08, high=0.08, size=(intermediate_dim, class_num)).astype('float32'), name='W2')
    b2 = tf.Variable(np.zeros(class_num).astype('float32'), name='b2')
    params = [W1, b1, W2, b2]

    # define graph
    u1 = tf.matmul(x, W1) + b1
    z1 = tf.nn.sigmoid(u1)
    u2 = tf.matmul(z1, W2) + b2
    y = tf.nn.softmax(u2)

    # define cost function
    cost = -tf.reduce_mean(tf.reduce_sum(t*tf.log(tf.clip_by_value(y, 1e-10, 1.0))))    

    # define parameters updating
    gW1, gb1, gW2, gb2 = tf.gradients(cost, params)
    updates = [
        W1.assign(W1 - eta * gW1),
        b1.assign(b1 - eta * gb1),
        W2.assign(W2 - eta * gW2),
        b2.assign(b2 - eta * gb2),
    ]

    # define training
    train = tf.group(*updates)

    # define one-hot -> integer value
    valid = tf.argmax(y, 1)


    # =========================================================================
    # Run Part
    # =========================================================================

    n_epochs = 100
    batch_size = train_X.shape[0] // 5
    n_batches = train_X.shape[0] // batch_size

    with tf.Session() as sess:
        # initialize tensor variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # train parameters
        for epoch in range(n_epochs):
            # batch learning
            train_X, train_y = shuffle(train_X, train_y, random_state=random_state)

            # batch processing
            for batch in range(n_batches):
                start = batch * batch_size
                end = start + batch_size
                sess.run(train, feed_dict={x: train_X[start:end], t: train_y[start:end]})
            pred_y, valid_cost = sess.run([valid, cost], feed_dict={x: valid_X, t: valid_y})
            print('EPOCH: {}, Validation cost: {:.3f}, Validation F1: {:.3f}'.format(epoch + 1, valid_cost, f1_score(np.argmax(valid_y, 1).astype('int32'), pred_y, average='macro')))

    print("END")
