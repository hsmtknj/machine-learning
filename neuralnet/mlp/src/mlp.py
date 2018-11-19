from sklearn.metrics import f1_score
from keras.datasets import mnist
from keras.utils import to_categorical

import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def deriv_sigmoid(x):
    return sigmoid(x)*(1 - sigmoid(x))

def softmax(x):
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x, axis=1, keepdims=True)

def deriv_softmax(x):
    return softmax(x)*(1 - softmax(x))

def tanh(x):
    return np.tanh(x)

def deriv_tanh(x):
    return 1 - np.tanh(x)**2


def load_preproc_data(data='mnist'):
    '''
    load training and test data and preprocess data
        :param data : str, name of data
        :return     : training and test data, input dimension, class number
    '''
    if data == 'mnist':
        (train_X, train_y), (test_X, test_y) = mnist.load_data()

        # normalize pixel value
        train_X = train_X.astype('float32') / 255.
        test_X = test_X.astype('float32') / 255.

        # (data number, 28, 28) -> (data number, 784)
        train_X = train_X.reshape((len(train_X), np.prod(train_X.shape[1:])))
        test_X = test_X.reshape((len(test_X), np.prod(test_X.shape[1:])))

        return train_X, train_y, test_X, test_y
    else:
        raise NotImplementedError


def train_mlp(train_X, train_y, test_X):

    # =========================================================
    #  set parameters
    # =========================================================
    input_dim = 784        # the number of pixel
    intermediate_dim = 50  # dimension of intermediate layer
    output_dim = 10        # 10 classes
    epoch = 5
    eps = 0.01
    
    
    # =========================================================
    #  Initialize parameters
    # =========================================================
    
    # Layer1 weights
    W1 = np.random.uniform(low=-0.08, high=0.08, size=(input_dim, intermediate_dim)).astype('float32')
    b1 = np.zeros(intermediate_dim).astype('float32')

    # Layer2 weights
    W2 = np.random.uniform(low=-0.08, high=0.08, size=(intermediate_dim, output_dim)).astype('float32')
    b2 = np.zeros(output_dim).astype('float32')

    
    # =========================================================
    #  Train
    # =========================================================
    # Epoch
    for epoch in range(epoch):
        print("epoch: " + str(epoch))

        # Online Learning
        for x, t in zip(train_X, train_y):
    
            # Forward Propagation Layer1
            u1 = np.matmul(x, W1) + b1
            z1 = 1/(1 + np.exp(-u1))  # sigmoid

            # Forward Propagation Layer2
            u2 = np.matmul(z1, W2) + b2
            z2 = 1/(1 + np.exp(-u2))  # sigmoid

            # Back Propagation (Cost Function: Negative Loglikelihood)
            y = z2
            tt = np.zeros(output_dim)
            tt[t] = 1  # one-of-k
            cost = np.sum(-tt*np.log(y) - (1 - tt)*np.log(1 - y))
            delta_2 = y - tt # Layer2 delta
            u1_deriv_sigmoid = 1/(1 + np.exp(-u1)) * (1 - 1/(1 + np.exp(-u1)))  # derivative sigmoid
            delta_1 = u1_deriv_sigmoid * np.matmul(delta_2, W2.T) # Layer1 delta

            # Update Parameters Layer1
            x = x.reshape((1, len(x)))
            delta_1 = delta_1.reshape(1, len(delta_1))
            dW1 = np.matmul(x.T, delta_1)
            db1 = np.matmul(np.ones(len(x)), delta_1)
            W1 = W1 - eps*dW1
            b1 = b1 - eps*db1

            # Update Parameters Layer2
            z1 = z1.reshape((1, len(z1)))
            delta_2 = delta_2.reshape(1, len(delta_2))            
            dW2 = np.matmul(z1.T, delta_2)
            db2 = np.matmul(np.ones(len(z1)), delta_2)
            W2 = W2 - eps*dW2
            b2 = b2 - eps*db2
    
    
    # =========================================================
    #  Predict
    # =========================================================
    
    # Forward Propagation Layer1
    u1 = np.matmul(test_X, W1) + b1
    z1 = 1/(1 + np.exp(-u1))  # sigmoid

    # Forward Propagation Layer2
    u2 = np.matmul(z1, W2) + b2
    z2 = 1/(1 + np.exp(-u2))  # sigmoid
    
    # translate one-of-k
    pred_y = np.argmax(z2, axis=1)
    
    return pred_y


if __name__ == '__main__':
    # load data
    print('start to load data')
    train_X, train_y, test_X, test_y = load_preproc_data()

    # train and predict
    print('start to train and predict')
    pred_y = train_mlp(train_X, train_y, test_X)
    # np.savetxt('./out.csv', pred_y, delimiter=',')

    # validate
    print(f1_score(test_y, pred_y, average='macro'))
