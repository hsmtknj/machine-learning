import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D

SW = 2

# =============================================================================
# laod iris datasets as pandas
# =============================================================================
#   [columns]
#       df['sepal length (cm)']
#       df['sepal width (cm)']
#       df['petal length (cm)']
#       df['petal width (cm)']
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target_names[iris.target]


# =============================================================================
# 3d scatter (1)
# =============================================================================

if (SW == 1):
    # set figure
    fig = plt.figure()
    ax = Axes3D(fig)

    # translate data frame into ndarray
    m = df.as_matrix()
    mx = df['sepal length (cm)'].as_matrix()
    my = df['sepal width (cm)'].as_matrix()
    mz = df['petal length (cm)'].as_matrix()

    # set labels and title
    ax.set_title('iris 3d plot')
    ax.set_xlabel('sepal length')
    ax.set_ylabel('sepal width')
    ax.set_zlabel('petal length')

    # set limits
    ax.set_xlim(min(mx), max(mx))
    ax.set_ylim(min(my), max(my))
    ax.set_zlim(min(mz), max(mz))

    # set colors
    clist = np.zeros(len(m))
    # print(clist)
    # print(clist.shape)

    # oters
    ax.grid(True)
    ax.legend(loc='upper left')

    # show figure
    ax.scatter(mx, my, mz, s=40, c='blue', marker='o', alpha=0.8, linewidths=1, edgecolors='blue', cmap='Blues')
    plt.show()


# =============================================================================
# 3d scatter (2)
# =============================================================================

if (SW == 2):
    # set figure
    fig = plt.figure()
    ax = Axes3D(fig)

    # translate data frame into ndarray
    m = df.as_matrix()
    m_s = df[df['target'] == 'setosa'].as_matrix()
    m_vi = df[df['target'] == 'virginica'].as_matrix()
    m_ve = df[df['target'] == 'versicolor'].as_matrix()

    # # get each columns index
    # col_ind = np.zeros(4)
    # col_ind[0] = df.columns.get_loc('sepal length (cm)')
    # col_ind[1] = df.columns.get_loc('sepal width (cm)')
    # col_ind[2] = df.columns.get_loc('petal length (cm)')
    # col_ind[3] = df.columns.get_loc('petal width (cm)')

    # # set labels and title
    ax.set_title('iris 3d plot')
    ax.set_xlabel('sepal length')
    ax.set_ylabel('sepal width')
    ax.set_zlabel('petal length')

    # set limits
    ax.set_xlim(min(m[:, 0]), max(m[:, 0]))
    ax.set_ylim(min(m[:, 1]), max(m[:, 1]))
    ax.set_zlim(min(m[:, 2]), max(m[:, 2]))

    # oters
    ax.grid(True)
    ax.legend(loc='upper left')

    # show figure
    ax.scatter(m_s[:, 0], m_s[:, 1], m_s[:, 2], s=40, c='blue', marker='o', alpha=1, linewidths=1, edgecolors='blue', cmap='Blues')
    ax.scatter(m_vi[:, 0], m_vi[:, 1], m_vi[:, 2], s=40, c='green', marker='s', alpha=1, linewidths=1, edgecolors='green', cmap='Greens')
    ax.scatter(m_ve[:, 0], m_ve[:, 1], m_ve[:, 2], s=40, c='red', marker='^', alpha=1, linewidths=1, edgecolors='red', cmap='Reds')
    plt.show()
