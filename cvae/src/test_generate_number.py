"""
[License]
test_generate_number.py

Copyright (c) 2018 Kenji Hashimoto

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php

[Module Description]
test training results (generate number)
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import model
from keras.utils import to_categorical
import os
from keras.models import load_model
import train


def gen_number(x_test, y_test, encoder, generator, dir_name='cvae/results/gen'):
    '''
    generating a number using a generator trained on MNIST and test data set.
        :param x_test    : ndarray, pictures of test data
        :param y_test    : ndarray, labels of test data
        :param encoder   : model object, encoder model to map a test data to latent space
        :param generator : model object, generator model
        :param dir_name  : str, path of directory to save generated pictures
    '''
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    n = 10           # size of class
    digit_size = 28  # pixel resolution

    figure = np.zeros((digit_size * n, digit_size * (n + 1)))

    for i in range(n):
        x = x_test[i]  # [1, 784]
        y = y_test[i]  # [1, 10]

        figure[i * digit_size: (i+1) * digit_size, 0: digit_size] = x.reshape(digit_size, digit_size)

        # z_encoded stand for charasterictic of input data
        z_encoded = encoder.predict([np.array([x]), np.array([y])])

        labels = np.array([np.zeros(shape=(n)) + x for x in range(n)])
        labels = to_categorical(labels)

        for j in range(len(labels)):
            label = labels[j]
            x_decoded = generator.predict([z_encoded, label])
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size, (j+1) * digit_size: (j + 2) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.savefig('{}/plot_gen'.format(dir_name))
    plt.close()


if __name__ == '__main__':
    # load model and generate number
    x_train, y_train, x_test, y_test, input_dim, class_num = train.load_preproc_data()
    encoder = load_model("cvae/results/model/encoder_model.h5")
    generator = load_model("cvae/results/model/generator_model.h5")
    gen_number(x_test, y_test, encoder, generator)    
