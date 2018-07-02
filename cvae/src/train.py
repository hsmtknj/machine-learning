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


def train(data, dir_name='results/model'):
    '''
    train with Conditional VAE on MNIST
        :param latent_dim : int, dimension of latent space
        :param data       : tuple of ndarray, training data and test data and dimensions
        :return: tuple of models, trained models (CVAE model, encoder model, decoder model)
    '''

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # =================================================
    # parameters setting
    # =================================================
    epochs = 20
    batch_size = 100
    latent_dim = 2
    intermediate_dim = 300
    dropout_keep_prob = 1.0


    # =================================================
    # training 
    # =================================================

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


if __name__ == '__main__':
    # load data and train
    data = x_train, y_train, x_test, y_test, input_dim, class_num = load_preproc_data()
    cvae_model, encoder, generator = train(data, 'results/model')
