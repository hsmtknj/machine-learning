import numpy as np
from keras.utils import to_categorical

x = np.zeros(shape=(10,1))
# print(x.shape)
# print(np.array([x]).shape)

n = 10
labels = np.array([np.zeros(shape=(n)) + x for x in range(n)])
labels = to_categorical(labels)

print(labels.shape)
print(labels)