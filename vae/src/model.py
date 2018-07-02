"""
[License]
model.py

Copyright (c) 2018 Kenji Hashimoto

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php

[Modele Description]
define model class (Variational AutoEncoder)
"""

from keras.models import Model
from keras.layers import Dense, Concatenate, Dropout, Input, Lambda, Layer
from keras import backend as K
from keras.metrics import binary_crossentropy


class ModelVAE(object):

    def __init__(self, input_dim, latent_dim=100, intermediate_dim=300, dropout_keep_prob=1.0):
        """
        Variational AutoEncoder(VAE) Class
            :param input_dim         : int, dimension of input (1d array)
            :param latent_dim        : int, dimension of latent variable
            :param intermediate_dim  : int, dimension of hidden layer in encoder and decoder
            :param dropout_keep_prob : float, probability of using a node (Default is 1.0, without Dropout)
        """
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.dropout_keep_prob = dropout_keep_prob

    def _sampling(self, args):
        """
        Reparameterization Trick, Sampling function for latent variable (using in Lambda layer)
            :param args            : tuple (mean, log sigma), parameters of Gaussian distribution for latent variable
            :return sampling_value : sampling value generated from Gaussian distribution
        """
        # gaussian parameters
        mean, log_sigma = args

        # generating random variable epsilon ~ N(0, I)
        #   shape=(None, dimension of latent_dim)
        epsilon = K.random_normal(shape=(K.shape(mean)[0], self.latent_dim), mean=0.,stddev=1.0)

        # return random variable z ( z ~ N(mean, exp(log_sigma)) )
        #   K.exp(log_sigma) * epsilon : This part is element-wise product
        sampling_value = mean + K.exp(log_sigma) * epsilon
        return sampling_value

    def _build_vae_model(self):
        """
        build vae model
            :return: vae_model, encoder_model, decoder_model
        """

        # =================================================
        # define layers
        # =================================================

        # define encoder layers
        enc_dense = Dense(self.intermediate_dim, activation='relu', name='enc_dense')
        enc_drop = Dropout(rate=self.dropout_keep_prob, name='enc_drop')
        enc_mean = Dense(self.latent_dim, activation='relu', name='enc_mean')
        enc_log_sigma = Dense(self.latent_dim, activation='relu', name='enc_log_sigma')

        # define decoder layers
        dec_dense = Dense(self.intermediate_dim, activation='relu', name='dec_dense')
        dec_out = Dense(self.output_dim, activation='sigmoid', name='dec_out')  # output layer(dense layer) using sigmoid. you should use use_bias=False


        # =================================================
        # build network architecture of cvae model
        # =================================================

        # ENCODER: INPUT LAYER
        # make input layer using Input() by concatenating two layers
        # NOTE: if you'd like to input actual data, you need to define a layer using Input()
        x = Input(shape=(self.input_dim,))  # features

        # ENCODER: HIDDEN LAYER
        enc_hidden_layer = enc_drop(enc_dense(x))
        self.mean_layer = enc_mean(enc_hidden_layer)
        self.log_sigma_layer = enc_log_sigma(enc_hidden_layer)

        # ENCODER: SAMPLING LAYER
        # get sample value using reparameterization trick for back propagation
        latent_layer = Lambda(self._sampling)([self.mean_layer, self.log_sigma_layer])

        # DECODER: HIDDEN LAYER
        dec_hidden_layer = dec_dense(latent_layer)

        # DECODER: OUTPUT LAYER
        output_layer = dec_out(dec_hidden_layer)


        # =================================================
        # define vae model and encoder model
        # =================================================

        vae_model = Model(x, output_layer)
        encoder_model = Model(x, self.mean_layer)


        # =================================================
        # define generator model
        # =================================================

        # network architecuture of generator model
        z = Input(shape=(self.latent_dim,))
        gen_hidden_layer = dec_dense(z)
        gen_output_layer = dec_out(gen_hidden_layer)

        # define generator model
        generator_model = Model(z, gen_output_layer)

        return vae_model, encoder_model, generator_model

    def _vae_loss(self, x, x_decoded):
        '''
        loss function for VAE
            :param x         : keras tensor object, input vector to be reconstructed.
            :param x_decoded : keras tensor object, output vector of decoder.
            :return: loss    : reconstruction error + kullback-leibler divergence
        '''
        reconstruction_error = self.input_dim * binary_crossentropy(x, x_decoded)
        kld = - 0.5 * K.sum(1 + self.log_sigma_layer - K.square(self.mean_layer) - K.exp(self.log_sigma_layer), axis=-1)

        # TODO: implementation of weight
        loss = reconstruction_error + kld
        return loss

    def get_simple_vae(self):
        '''
        build a simple VAE
            :return: tuple of models, (CVAE model, encoder model, decoder model)
        '''
        vae_model, encoder_model, generator_model = self._build_vae_model()
        vae_model.compile(optimizer='rmsprop', loss=self._vae_loss)

        return vae_model, encoder_model, generator_model
    