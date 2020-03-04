"""
Model Class Definition
"""

# register src directory path to PYTHONPATH
import sys
from os import path, pardir
current_dir = path.abspath(path.dirname(__file__))
parent_dir = path.abspath(path.join(current_dir, pardir))
parent_parent_dir = path.abspath(path.join(parent_dir, pardir))
sys.path.append(parent_dir)

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from modeldev import d_layer
from mylib import utils

rng = np.random.RandomState(1000)
random_state = 50


class Model(object):
    """
    Neural Network Model
        In this model, the following networks is implemented.
            - VAE: Variational AutoEncoder
            - MLP: Simple Multilayer Perceptron
        VAE is used to encord input features.
        Simple MLP is used to output.
            Input of simple MLP is an input which is encoded with VAE.
    """
    
    def __init__(self,
                 vae_layer_dim_list,
                 vae_layer_type_list,
                 vae_act_func_list,
                 vae_loss_func,
                 vae_optimizer,
                 vae_epochs,
                 vae_kld_coef,
                 mlp_layer_dim_list,
                 mlp_layer_type_list,
                 mlp_act_func_list,
                 mlp_loss_func,
                 mlp_optimizer,
                 mlp_epochs):
                 """
                 Network initialization
                 """

                 self.latent_dim = vae_layer_dim_list[len(vae_layer_dim_list) - 1]

                 # set vae model variables
                 self.vae_layer_dim_list = vae_layer_dim_list
                 self.vae_layer_type_list = vae_layer_type_list
                 self.vae_act_func_list = vae_act_func_list
                 self.vae_loss_func = vae_loss_func
                 self.vae_optimizer = vae_optimizer
                 self.vae_epochs = vae_epochs
                 vae_layer_last_ind = len(vae_layer_dim_list) - 1
                 self.vae_kld_coef = vae_kld_coef

                 # set mlp model variables
                 self.mlp_layer_dim_list = list([ vae_layer_dim_list[vae_layer_last_ind] ]) + mlp_layer_dim_list
                 self.mlp_layer_type_list = mlp_layer_type_list
                 self.mlp_act_func_list = mlp_act_func_list
                 self.mlp_loss_func = mlp_loss_func
                 self.mlp_optimizer = mlp_optimizer
                 self.mlp_epochs = mlp_epochs

                 # build model
                 self._build_model()

    def _build_model(self):
        """
        build network model
        """

        self.x = tf.placeholder(dtype=np.dtype('float32'), shape=[None, self.vae_layer_dim_list[0]])
        self.t = tf.placeholder(dtype=np.dtype('float32'), shape=[None, self.mlp_layer_dim_list[len(self.mlp_layer_dim_list) - 1]])

        self._build_vae_model()
        self._build_mlp_model()


    def _build_vae_model(self):
        """
        build Variationnal AutoEncoder model
        """
        
        # initialize encode constant layer
        #   This will be updated later when fitting with VAE is finished
        #   and used in the case of "transfer learning".
        #   ※ In the case of "fine tuning", this will not be used.
        self.W_enc_const = []
        self.b_enc_const = []
        for i in range(len(self.vae_layer_type_list)):
            W_enc = np.zeros(shape=(self.vae_layer_dim_list[i], self.vae_layer_dim_list[i+1]), dtype=np.dtype('float32'))
            b_enc = np.zeros(shape=(self.vae_layer_dim_list[i+1]), dtype=np.dtype('float32'))
            self.W_enc_const.append(tf.constant(W_enc, dtype='float32'))
            self.b_enc_const.append(tf.constant(b_enc, dtype='float32'))

        # define encoder layers
        self.encoder_layers = []
        encoder_layer_dim_list = self.vae_layer_dim_list
        encoder_layer_type_list = self.vae_layer_type_list
        encoder_act_func_list = self.vae_act_func_list
        for i, (encoder_layer, encoder_act_func) in enumerate(zip(encoder_layer_type_list, encoder_act_func_list)):
            # mean layers (NOTE: list)
            self.encoder_layers.append(encoder_layer(encoder_layer_dim_list[i],
                                                     encoder_layer_dim_list[i+1],
                                                     encoder_act_func,
                                                     'W_' + 'e' + str(i),
                                                     'b_' + 'e' + str(i)))

            # sigma layer (NOTE: NOT list)
            if (i == len(encoder_layer_type_list) - 1):
                self.sigma_layer = encoder_layer(encoder_layer_dim_list[i],
                                                 encoder_layer_dim_list[i+1],
                                                 encoder_act_func,
                                                 'W_' + 'sigma',
                                                 'b_' + 'sigma')

        # define decoder layers
        self.decoder_layers = []
        decoder_layer_dim_list = encoder_layer_dim_list
        decoder_layer_dim_list.reverse()
        decoder_layer_type_list = encoder_layer_type_list
        decoder_layer_type_list.reverse()
        decoder_act_func_list = encoder_act_func_list
        decoder_act_func_list.reverse()
        for i, (decoder_layer, decoder_act_func) in enumerate(zip(decoder_layer_type_list, decoder_act_func_list)):
            self.decoder_layers.append(decoder_layer(decoder_layer_dim_list[i],
                                                     decoder_layer_dim_list[i+1],
                                                     decoder_act_func,
                                                     'W_' + 'd' + str(i),
                                                     'b_' + 'd' + str(i)))

        # define graph
        self.mu, self.log_sigma = self._encode(self.x)

        # HACK: Now epsilon is not "self.epsilon". Is it OK?
        #       Review initialization number
        epsilon = tf.Variable(rng.normal(loc=0.0, scale=1.0, size=(self.latent_dim)).astype('float32'), name='epsilon')
        self.z = self.mu + tf.exp(self.log_sigma) * epsilon  # reparametrization trick

        self.reconst_x = self._decode(self.z)

        # TODO:
        # [] save vae_reconst_loss
        # [] save kld
        # [] save vae_loss
        # [] consder coefficient of "kld"
        self.vae_reconst_loss = self.vae_loss_func(self.x, self.reconst_x)  # reconstruction error
        self.kld = tf.reduce_mean(1 / 2 * (-2 * self.log_sigma + tf.exp(self.log_sigma)**2 + self.mu**2 - 1))  # kl divergence
        self.vae_loss = self.vae_reconst_loss + self.vae_kld_coef * self.kld
        self.vae_train = self.vae_optimizer().minimize(self.vae_loss)
        

    def _build_mlp_model(self):
        """
        build Simple Multilayer Perceptron Model
        """
        
        # define MLP layers
        self.mlp_layers = []
        for i, (mlp_layer, mlp_act_func) in enumerate(zip(self.mlp_layer_type_list, self.mlp_act_func_list)):
            self.mlp_layers.append(mlp_layer(self.mlp_layer_dim_list[i],
                                             self.mlp_layer_dim_list[i+1],
                                             mlp_act_func,
                                             'W_' + 'm' + str(i),
                                             'b_' + 'm' + str(i)))

        # encode input with parameters which VAE model outputs
        # TODO: implement "switch" or something to change "transfer learning" or "fine tuning"
        self.encoded_x = self._encode_const_param(self.x)  # transfer learning
        # self.encoded_x = self._encode(self.x)  # fine tuning

        # define graph
        u = self.encoded_x
        for i, mlp_layer in enumerate(self.mlp_layers):
            u = mlp_layer.f_prop(u)
        self.y = u

        self.mlp_loss = self.mlp_loss_func(self.t, self.y)
        self.mlp_train = self.mlp_optimizer().minimize(self.mlp_loss)


    def _encode(self, x):
        """
        encoder of VAE
        """
        u = x
        for i, encode_layer in enumerate(self.encoder_layers):
            if (i == (len(self.encoder_layers) - 1)):
                log_sigma = self.sigma_layer.f_prop(u)
            u = encode_layer.f_prop(u)
        mu = u

        return mu, log_sigma


    def _decode(self, z):
        """
        decoder of VAE
        """
        u = z
        for i, decoder_layer in enumerate(self.decoder_layers):
            u = decoder_layer.f_prop(u)
        decoded_x = u
        return decoded_x


    def _keep_encoder_param(self, sess):
        """
        keep encoder parameters when you use transfer learning
        """
        for i in range(len(self.vae_layer_type_list)):
            W_enc = sess.run(self.encoder_layers[i].W)
            b_enc = sess.run(self.encoder_layers[i].b)
            self.W_enc_const[i] = tf.constant(W_enc, dtype='float32')
            self.b_enc_const[i] = tf.constant(b_enc, dtype='float32')
        return sess


    def _encode_const_param(self, x):
        """
        encode input feature with constant parameters of encoder for transfer learning
        """
        u = x
        for i in range(len(self.W_enc_const)):
            u = tf.matmul(u, self.W_enc_const[i]) + self.b_enc_const[i]
        z = u
        return z


    def vae_fit(self, sess, train_X, valid_X, flag_error_disp=True):
        """
        fit parameters of vae model
        """

        for epoch in range(self.vae_epochs):
            sess.run(self.vae_train, feed_dict={self.x:train_X})

            if (flag_error_disp is True):
                vae_error = sess.run(self.vae_loss, feed_dict={self.x:valid_X})
                vae_reconst_error = sess.run(self.vae_reconst_loss, feed_dict={self.x:valid_X})
                kld = sess.run(self.kld, feed_dict={self.x:valid_X})
                print('EPOCH:{}, VAE ERROR:{} (RECONST:{}, KLD:{})'.format(epoch+1, vae_error, vae_reconst_error, kld))
        
        return sess


    def mlp_fit(self, sess, train_X, train_y, valid_X, valid_y, flag_error_disp=True):
        """
        fit parameters of mlp model
        """
        for epoch in range(self.mlp_epochs):
            sess.run(self.mlp_train, feed_dict={self.x:train_X, self.t:train_y})

            if (flag_error_disp is True):
                mlp_error = sess.run(self.mlp_loss, feed_dict={self.x:valid_X, self.t:valid_y})
                print('EPOCH:{}, MLP ERROR:{}'.format(epoch+1, mlp_error))

        return sess


    def fit(self, sess, train_X, train_y, valid_X, valid_y):
        """
        fit parameters
        """

        # pre training with VAE for extracting important input feature
        sess = self.vae_fit(sess, train_X, valid_X, True)
        sess = self._keep_encoder_param(sess)

        # training with MLP using part of VAE layers
        sess = self.mlp_fit(sess, train_X, train_y, valid_X, valid_y, True)

        return sess


    def predict(self, sess, test_X):
        """
        predicttion
        """
        pred = sess.run(self.y, feed_dict={self.x:test_X})

        return pred
    