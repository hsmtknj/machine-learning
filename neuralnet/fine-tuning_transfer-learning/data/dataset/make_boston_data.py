from sklearn.datasets import load_boston
import pandas as pd

def make_boston_data():
    boston = load_boston()

    print(boston.data.shape)
    print(boston.target.shape)

    data_X = pd.DataFrame(boston.data, columns=boston.feature_names)
    data_y = pd.DataFrame(boston.target)

    data_X.to_csv('/Users/khashimoto/Programming/git/ml/neuralnet/fine-tuning_transfer-learning/data/dataset/data_X.csv', header=False, index=False)
    data_y.to_csv('/Users/khashimoto/Programming/git/ml/neuralnet/fine-tuning_transfer-learning/data/dataset/data_y.csv', header=False, index=False)


if __name__ == '__main__':
    make_boston_data()
