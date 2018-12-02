
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import os
import cv2

random_state = 42

if __name__ == '__main__':

    # =========================================================================
    # Load Data (mnist)
    # =========================================================================

    # load data and reshape
    #   train_X, test_X : (60000, 784)
    #   train_y, test_y : (60000, )
    mnist = tf.keras.datasets.mnist
    (train_X, train_y),(test_X, test_y) = mnist.load_data()
    train_X, test_X = train_X / 255.0, test_X / 255.0
    train_X = train_X.reshape(train_X.shape[0], train_X.shape[1]*train_X.shape[2])
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[1]*test_X.shape[2])

    # convert small datasets
    train_X = train_X[0:100]
    test_X = test_X[0:100]
    train_y = train_y[0:100]
    test_y = test_y[0:100]
    

    # =========================================================================
    # Hyper Parameters
    # =========================================================================
    
    INPUT_DIM = 784
    HIDDEN_DIM = 300
    LATENT_DIM = 2
    OUTPUT_DIM = INPUT_DIM

    EPOCHS = 500
    N_BATCHES = 5
    BATCH_SIZE = train_X.shape[0] // N_BATCHES
    LEARNING_RATIO = 0.0001


    # =========================================================================
    # Define Part
    # =========================================================================

    # input data
    x = tf.placeholder(tf.float32, [None, INPUT_DIM])

    # -------------------------------------------------------------------------
    # layer parameters
    # -------------------------------------------------------------------------

    # [Encoder] input layer
    W_encoder = tf.Variable((np.random.randn(INPUT_DIM, HIDDEN_DIM) / np.sqrt(INPUT_DIM)).astype('float32'))
    b_encoder = tf.Variable(np.random.randn(HIDDEN_DIM).astype('float32'))

    # [Encoder] mu and sigma layer
    W_mu = tf.Variable((np.random.randn(HIDDEN_DIM, LATENT_DIM) / np.sqrt(HIDDEN_DIM)).astype('float32'))
    b_mu = tf.Variable(np.random.randn(LATENT_DIM).astype('float32'))

    W_sigma = tf.Variable((np.random.randn(HIDDEN_DIM, LATENT_DIM) / np.sqrt(HIDDEN_DIM)).astype('float32'))
    b_sigma = tf.Variable(np.random.randn(LATENT_DIM).astype('float32'))

    # [Decoder] output layer
    W_decoder1 = tf.Variable((np.random.randn(LATENT_DIM, HIDDEN_DIM) / np.sqrt(LATENT_DIM)).astype('float32'))
    b_decoder1 = tf.Variable(np.random.randn(HIDDEN_DIM).astype('float32'))

    W_decoder2 = tf.Variable((np.random.randn(HIDDEN_DIM, OUTPUT_DIM) / np.sqrt(HIDDEN_DIM)).astype('float32'))
    b_decoder2 = tf.Variable(np.random.randn(OUTPUT_DIM).astype('float32'))


    # -------------------------------------------------------------------------
    # forward propagation
    # -------------------------------------------------------------------------

    # [Encoder] input layer
    node_encoder_hidden = tf.matmul(x, W_encoder) + b_encoder
    node_encoder_hidden_activated = tf.nn.relu(node_encoder_hidden)

    # W_tmp = tf.Variable(np.random.randn(300, 784).astype('float32'))
    # b_tmp = tf.Variable(np.random.randn(784).astype('float32'))

    # y = tf.matmul(node_encoder_hidden_activated, W_tmp) + b_tmp
    # reconst_x = tf.sigmoid(y)

    # [Encoder] mu and sigma layer
    mu = tf.matmul(node_encoder_hidden_activated, W_mu) + b_mu
    log_sigma = tf.matmul(node_encoder_hidden_activated, W_sigma) + b_sigma

    # latent variable z
    epsilon = tf.random.normal([LATENT_DIM], mean=0.0, stddev=1.0, dtype=tf.float32)
    z = mu + tf.exp(log_sigma) * epsilon

    # [Decoder] hidden layer
    node_decoder_hidden = tf.matmul(z, W_decoder1) + b_decoder1
    node_decoder_hidden_activated = tf.nn.relu(node_decoder_hidden)

    # [Decoder] output layer
    node_output = tf.matmul(node_decoder_hidden_activated, W_decoder2) + b_decoder2
    reconst_x = tf.sigmoid(node_output)

    # loss function (cross entropy)
    loss = -tf.reduce_mean(tf.reduce_sum(x * tf.log(reconst_x) + (1. - x) * tf.log(1. - reconst_x), axis=1))

    # optimizer
    train = tf.train.GradientDescentOptimizer(LEARNING_RATIO).minimize(loss)

    

    # =========================================================================
    # Run Part
    # =========================================================================

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # training
        for epoch in range(EPOCHS):
            # shuffle data
            train_X, train_y = shuffle(train_X, train_y, random_state=random_state)

            # batch learning
            for i in range(N_BATCHES):
                start = BATCH_SIZE * i
                end = start + BATCH_SIZE
                sess.run(train, feed_dict={x: train_X[start:end]})
            error = sess.run(loss, feed_dict={x: test_X})
            print("EPOCH: {}, ERROR: {}".format(epoch + 1, error))
        
        # reconstruction
        print(train_X[0].shape)
        img_reconst = sess.run(reconst_x, feed_dict={x: train_X[0].reshape(1, 784)})

    img_reconst = img_reconst.reshape(28, 28)
    img_origin = train_X[0].reshape(28, 28)
    cv2.imshow("Original Image", img_origin)
    cv2.imshow("Reconstruction Image", img_reconst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
