import sys
from os import path, pardir
current_dir = path.abspath(path.dirname(__file__))
parent_dir = path.abspath(path.join(current_dir, pardir))
parent_parent_dir = path.abspath(path.join(parent_dir, pardir))
sys.path.append(parent_dir)

from sklearn.datasets import load_boston
import pandas as pd

def make_boston_data():
    boston = load_boston()

    print(boston.data.shape)
    print(boston.target.shape)

    data_X = pd.DataFrame(boston.data, columns=boston.feature_names)
    data_y = pd.DataFrame(boston.target)

    dirname = parent_parent_dir + '/data/dataset/'
    data_X.to_csv(dirname + 'data_X.csv', header=False, index=False)
    data_y.to_csv(dirname + 'data_y.csv', header=False, index=False)


if __name__ == '__main__':
    make_boston_data()
