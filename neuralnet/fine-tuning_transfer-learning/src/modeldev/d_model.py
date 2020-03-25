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
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from modeldev import d_layer
from mylib import utils

rng = np.random.RandomState(1000)
random_state = 4


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
                 network initialization
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

                 # set others
                 self.flag_error_disp = False
                 self.disp_step = 1
                 self.learning_method = 'transfer-learning'

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

        self.vae_reconst_loss = self.vae_loss_func(self.x, self.reconst_x)  # reconstruction error
        self.kld = tf.reduce_mean(1 / 2 * (-2 * self.log_sigma + tf.exp(self.log_sigma)**2 + self.mu**2 - 1))  # kl divergence
        self.vae_loss = self.vae_reconst_loss + self.vae_kld_coef * self.kld
        self.vae_train = self.vae_optimizer().minimize(self.vae_loss)
        

    def _build_mlp_model(self):
        """
        build Simple Multilayer Perceptron Model that an input is latent of VAE
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
        # select "transfer learning" or "fine tuning"
        if (self.learning_method == 'transfer-learning'):
            self.encoded_x = self._encode_const_param(self.x)
        elif (self.learning_method == 'fine-tuning'):
            self.encoded_x = self._encode(self.x)

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

        :param  x: tensor
        :return
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

        :param  z: tensor
        :return
        """
        u = z
        for i, decoder_layer in enumerate(self.decoder_layers):
            u = decoder_layer.f_prop(u)
        decoded_x = u
        return decoded_x


    def _keep_encoder_param(self, sess):
        """
        keep encoder parameters when you use transfer learning

        :param  sess: tf.Session()
        :return
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

        :param  x: tensor
        :return
        """
        u = x
        for i in range(len(self.W_enc_const)):
            u = tf.matmul(u, self.W_enc_const[i]) + self.b_enc_const[i]
        z = u
        return z


    def vae_fit(self, sess, train_X, valid_X):
        """
        fit parameters of vae model

        :param  sess   : tf.Session()
        :param  train_X: ndarray
        :param  valid_X: ndarray
        """
        # initialize error list
        epoch_list = []
        vae_error_list = []
        vae_reconst_error_list = []
        kld_list = []

        vae_reconst_error_mse_test_list = []
        vae_reconst_error_mae_test_list = []
        vae_reconst_error_r2_test_list = []

        vae_reconst_error_mse_train_list = []
        vae_reconst_error_mae_train_list = []
        vae_reconst_error_r2_train_list = []

        for epoch in range(self.vae_epochs):
            #* Train *#
            sess.run(self.vae_train, feed_dict={self.x:train_X})

            #* Calculate Error *#
            epoch_list.append(epoch)

            # vae error
            vae_error = sess.run(self.vae_loss, feed_dict={self.x:valid_X})
            vae_reconst_error = sess.run(self.vae_reconst_loss, feed_dict={self.x:valid_X})
            kld = sess.run(self.kld, feed_dict={self.x:valid_X})

            vae_error_list.append(vae_error)
            vae_reconst_error_list.append(vae_reconst_error)
            kld_list.append(kld)

            # test error
            reconst_x_test = sess.run(self.reconst_x, feed_dict={self.x:valid_X})
            vae_reconst_error_mse_test = mean_squared_error(valid_X, reconst_x_test)
            vae_reconst_error_mae_test = mean_absolute_error(valid_X, reconst_x_test)
            vae_reconst_error_r2_test = r2_score(valid_X, reconst_x_test)

            vae_reconst_error_mse_test_list.append(vae_reconst_error_mse_test)
            vae_reconst_error_mae_test_list.append(vae_reconst_error_mae_test)
            vae_reconst_error_r2_test_list.append(vae_reconst_error_r2_test)

            # train error
            reconst_x_train = sess.run(self.reconst_x, feed_dict={self.x:train_X})
            vae_reconst_error_mse_train = mean_squared_error(train_X, reconst_x_train)
            vae_reconst_error_mae_train = mean_absolute_error(train_X, reconst_x_train)
            vae_reconst_error_r2_train = r2_score(train_X, reconst_x_train)

            vae_reconst_error_mse_train_list.append(vae_reconst_error_mse_train)
            vae_reconst_error_mae_train_list.append(vae_reconst_error_mae_train)
            vae_reconst_error_r2_train_list.append(vae_reconst_error_r2_train)

            if (self.flag_error_disp is True
                and (   (epoch % self.disp_step == 0)
                     or (epoch == self.mlp_epochs-1))):
                print('EPOCH:{}, VAE ERROR:{} (RECONST:{}, KLD:{})'.format(epoch+1, vae_error, vae_reconst_error, kld))
            
        # save epoch-error results
        self.vae_epoch_error = pd.DataFrame({'epoch': epoch_list,
                                             'vae_error': vae_error_list,
                                             'vae_reconst_error': vae_reconst_error_list,
                                             'kld': kld_list,
                                             'vae_reconst_error_mse_test': vae_reconst_error_mse_test_list,
                                             'vae_reconst_error_mae_test': vae_reconst_error_mae_test_list,
                                             'vae_reconst_error_r2_test': vae_reconst_error_r2_test_list,
                                             'vae_reconst_error_mse_train': vae_reconst_error_mse_train_list,
                                             'vae_reconst_error_mae_train': vae_reconst_error_mae_train_list,
                                             'vae_reconst_error_r2_train': vae_reconst_error_r2_train_list})
        
        return sess


    def mlp_fit(self, sess, train_X, train_y, valid_X, valid_y):
        """
        fit parameters of mlp model

        :param  sess   : tf.Session()
        :param  train_X: ndarray
        :param  train_y: ndarray
        :param  valid_X: ndarray
        :param  valid_y: ndarray
        """
        # initialize error list
        epoch_list = []
        mlp_error_list = []

        mlp_error_mse_test_list = []
        mlp_error_mae_test_list = []
        mlp_error_r2_test_list = []

        mlp_error_mse_train_list = []
        mlp_error_mae_train_list = []
        mlp_error_r2_train_list = []

        for epoch in range(self.mlp_epochs):
            #* Train *#
            sess.run(self.mlp_train, feed_dict={self.x:train_X, self.t:train_y})

            #* Calculate Error *#
            epoch_list.append(epoch)
            
            # mlp error
            mlp_error = sess.run(self.mlp_loss, feed_dict={self.x:valid_X, self.t:valid_y})
            mlp_error_list.append(mlp_error)

            # # test error
            pred_y_test = sess.run(self.y, feed_dict={self.x:valid_X})
            mlp_error_mse_test = mean_squared_error(valid_y, pred_y_test)
            mlp_error_mae_test = mean_absolute_error(valid_y, pred_y_test)
            mlp_error_r2_test = r2_score(valid_y, pred_y_test)

            mlp_error_mse_test_list.append(mlp_error_mse_test)
            mlp_error_mae_test_list.append(mlp_error_mae_test)
            mlp_error_r2_test_list.append(mlp_error_r2_test)

            # # train error
            pred_y_train = sess.run(self.y, feed_dict={self.x:train_X})
            mlp_error_mse_train = mean_squared_error(train_y, pred_y_train)
            mlp_error_mae_train = mean_absolute_error(train_y, pred_y_train)
            mlp_error_r2_train = r2_score(train_y, pred_y_train)

            mlp_error_mse_train_list.append(mlp_error_mse_train)
            mlp_error_mae_train_list.append(mlp_error_mae_train)
            mlp_error_r2_train_list.append(mlp_error_r2_train)

            if (self.flag_error_disp is True
                and (   (epoch % self.disp_step == 0)
                     or (epoch == self.mlp_epochs-1))):
                print('EPOCH:{}, MLP ERROR:{}'.format(epoch+1, mlp_error))

        # save epoch-error results
        self.mlp_epoch_error = pd.DataFrame({'epoch': epoch_list,
                                             'mlp_error': mlp_error_list,
                                             'mlp_error_mse_test': mlp_error_mse_test_list,
                                             'mlp_error_mae_test': mlp_error_mae_test_list,
                                             'mlp_error_r2_test': mlp_error_r2_test_list,
                                             'mlp_error_mse_train': mlp_error_mse_train_list,
                                             'mlp_error_mae_train': mlp_error_mae_train_list,
                                             'mlp_error_r2_train': mlp_error_r2_train_list})

        return sess


    def fit(self, sess, train_X, train_y, valid_X, valid_y):
        """
        fit parameters with VAE and MLP

        :param  sess   : tf.Session()
        :param  train_X: ndarray
        :param  train_y: ndarray
        :param  valid_X: ndarray
        :param  valid_y: ndarray
        """
        # pre training with VAE for extracting important input feature
        sess = self.vae_fit(sess, train_X, valid_X)
        sess = self._keep_encoder_param(sess)

        # training with MLP using part of VAE layers
        sess = self.mlp_fit(sess, train_X, train_y, valid_X, valid_y)

        return sess


    def predict(self, sess, test_X):
        """
        predicttion

        :param  sess  : tf.Session()
        :param  test_X: ndarray
        :return
        """
        pred = sess.run(self.y, feed_dict={self.x:test_X})

        return pred
    

    def set_train_opt(self, **kwargs):
        """
        set training option

        :param  in_flag_error_disp: bool, error diplay
        :param  disp_step         : int, display error every "disp step"
        :param  learning_method   : str, "transfer-learning" or "fine-tuning"
        :return 
        """
        if ('flag_error_disp' in kwargs):
            self.flag_error_disp = kwargs['flag_error_disp']
        if ('disp_step' in kwargs):
            self.disp_step = kwargs['disp_step']
        if ('learning_method' in kwargs):
            self.learning_method = kwargs['learning_method']
