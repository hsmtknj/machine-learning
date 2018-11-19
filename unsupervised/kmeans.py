"""
unsupervised learning algorithm

implement k-means algorithm
"""
import numpy as np
import matplotlib.pyplot as plt

# TODO: implement k-means algorithm
#   Based algorithm is implemented in tmp.py.
#   So you need to convert and move source code into this script from tmp.py.s
if (__name__ == '__main__'):
    # define data using pseud random numbers
    np.random.seed(100)
    data1 = np.random.normal(20, 10, 300)
    data2 = np.random.normal(100, 20, 300)

    # set figure
    print(data1.shape)
    print(data2.shape)
    fig1 = plt
    fig2 = plt
    fig1.hist(data1, bins=25, normed=True)
    fig2.hist(data2, bins=25, normed=True)
    fig2.show()
