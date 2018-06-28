"""
[License]
train.py

Copyright (c) 2018 Kenji Hashimoto

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php

[Module Description]
train with model
"""

from keras.datasets import mnist
import numpy as np
import model
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import os


def load_preproc_data(data='mnist'):
    '''
    load training and test data and tailor data for CVAE
        :param data : str, name of data
        :return     : training and test data, input dimension, class number
    '''
    if data == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # normalize pixel value
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.

        # (data number, 28, 28) -> (data number, 784)
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

        # e.g) 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        class_num = 10
        input_dim = 784

        return x_train, y_train, x_test, y_test, input_dim, class_num
    else:
        raise NotImplementedError


def train(latent_dim, data, dir_name='results/model'):
    '''
    train with Conditional VAE on MNIST
        :param latent_dim : int, dimension of latent space
        :param data       : tuple of ndarray, training data and test data and dimensions
        :return: tuple of models, trained models (CVAE model, encoder model, decoder model)
    '''

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # =================================================
    # training 
    # =================================================

    # parameters setting
    epochs = 20
    batch_size = 100

    # define model and fit
    x_train, y_train, x_test, y_test, input_dim, class_num = data
    model_instance = model.ModelCVAE(class_num, input_dim, latent_dim)
    cvae_model, encoder_model, generator_model = model_instance.get_simple_cvae()

    cvae_model.fit([x_train, y_train], x_train,
                   shuffle=True,
                   epochs=epochs,
                   batch_size=batch_size,
                   validation_data=([x_test, y_test], x_test))


    # =================================================
    # save model
    # =================================================

    # save cvae models and weights
    cvae_model.save('{}/{}.h5'.format(dir_name, 'cvae_model'))
    cvae_model.save_weights('{}/{}.h5'.format(dir_name, 'cvae_model_weight'))

    # set cvae model weights to encder and generator model
    # This part is unnecessary, isn't it?
    encoder_model.load_weights('{}/{}.h5'.format(dir_name, 'cvae_model_weight'), by_name=True)
    generator_model.load_weights('{}/{}.h5'.format(dir_name, 'cvae_model_weight'), by_name=True)

    # save encoder and generator model
    encoder_model.save('{}/{}.h5'.format(dir_name, 'encoder_model'))
    encoder_model.save_weights('{}/{}.h5'.format(dir_name, 'encoder_model_weight'))
    generator_model.save('{}/{}.h5'.format(dir_name, 'generator_model'))
    generator_model.save_weights('{}/{}.h5'.format(dir_name, 'generator_model_weight'))

    return cvae_model, encoder_model, generator_model


def plot_manifold(generator, dir_name='results/manifold'):
    '''
    plot a number on 2D manifold
        :param generator : model object, model of generator
        :param dir_name  : str, path of directory for saving figs
    '''
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    epsilon_std = 0.2

    # display a 2D manifold of the digits
    n = 15  # figure with 15x15 digits
    digit_size = 28

    figure = np.zeros((digit_size * n, digit_size * n))
    # we will sample n points within [-15, 15] standard deviations
    grid_x = np.linspace(-15, 15, n)
    grid_y = np.linspace(-15, 15, n)

    labels = np.array([np.zeros(shape=(n * n)) + x for x in range(10)])
    labels = to_categorical(labels)

    for i in range(len(labels)):
        label = labels[i]
        for j, yi in enumerate(grid_x):
            for k, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]]) * epsilon_std
                x_decoded = generator.predict([z_sample, label])
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[j * digit_size: (j + 1) * digit_size,
                k * digit_size: (k + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure)
        plt.savefig('{}/plot_{}'.format(dir_name, i))
        plt.close()

    z2 = labels[7] + labels[9]
    for j, yi in enumerate(grid_x):
        for k, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]]) * epsilon_std
            x_decoded = generator.predict([z_sample, z2])
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[j * digit_size: (j + 1) * digit_size,
            k * digit_size: (k + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.savefig('{}/plot_{}and{}'.format(dir_name, 7, 9))
    plt.close()


def gen_number(x_test, y_test, encoder, generator, dir_name='results/gen'):
    '''
    generating a number using a generator trained on MNIST and test data set.
    :param x_test: ndarray, pictures of test data
    :param y_test: ndarray, labels of test data
    :param encoder: model object, encoder model to map a test data to latent space
    :param generator: model object, generator model
    :param dir_name: str, path of directory to save generated pictures
    '''
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    n = 10
    digit_size = 28

    figure = np.zeros((digit_size * n, digit_size * (n + 1)))

    for i in range(n):
        x = x_test[i]
        y = y_test[i]

        figure[i * digit_size: (i+1) * digit_size, 0: digit_size] = x.reshape(digit_size, digit_size)

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

    latent_dim = 2
    data = x_train, y_train, x_test, y_test, input_dim, class_num = load_preproc_data()
    cvae_model, encoder, generator = train(latent_dim, data)

    plot_manifold(generator)

    gen_number(x_test, y_test, encoder, generator)
