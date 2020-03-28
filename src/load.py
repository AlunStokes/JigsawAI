import pickle
import os

from skimage.io import imread

import numpy as np

def load_train_data(path):

    X = [imread(os.path.join(path, x)) for x in os.listdir(path) if x[0] != '.']
    i = 0
    while i < len(X):
        if len(X[i].shape) == 2:
            X[i] = np.stack((X[i],)*3, axis=-1)
        i += 1
    X = np.array(X)
    X = X/np.float32(255)

    return X

    img_size = int((len(X[0]) // 3)**(1/2))
    img_size2 = img_size * img_size

    X = np.dstack((X[:, :img_size2], X[:, img_size2:2*img_size2], X[:, 2*img_size2:]))
    X = X.reshape((X.shape[0], img_size, img_size, 3)).transpose(0, 1, 2, 3)
    return X
