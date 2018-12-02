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


def plot_manifold(generator, dir_name='vae/results/manifold'):
    '''
    plot a number on 2D manifold
        :param generator: model object, model of generator
        :param dir_name : str, path of directory for saving figs
    '''
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    epsilon_std = 0.2

    # setting figure size to display a 2D manifold
    n = 50           # figure with n x n digits
    digit_size = 28  # resolution size of image
    figure = np.zeros((digit_size * n, digit_size * n))

    # setting grid size to sample n points within [grid_min grid_max] standart deviations
    grid_max = n
    grid_min = -n
    grid_x = np.linspace(grid_min, grid_max, n)
    grid_y = np.linspace(grid_min, grid_max, n)

    # make figure
    for j, yi in enumerate(grid_x):
        for k, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]]) * epsilon_std
            x_decoded = generator.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[j * digit_size: (j + 1) * digit_size,
            k * digit_size: (k + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure)
        plt.savefig('{}/plot'.format(dir_name))
        plt.close()


if __name__ == '__main__':
    # load model and generate number
    x_train, y_train, x_test, y_test, input_dim, class_num = train.load_preproc_data()
    generator = load_model("vae/results/model/generator_model.h5")
    plot_manifold(generator)
