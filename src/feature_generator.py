import random

import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.models import Sequential

from utilities import *
from model import network

def generate_match_horiz(x):
    s = x.shape[0] // 2
    p1 = x[0:s, 0:s, :]
    p2 = x[0:s, s:, :]
    p3 = x[s:, 0:s, :]
    p4 = x[s:, s:, :]

    if np.random.rand() < 0.5:
        return np.hstack([p1, p2])
    return np.hstack([p3, p4])

def generate_match_vert(x):
    s = x.shape[0] // 2
    p1 = x[0:s, 0:s, :]
    p2 = x[0:s, s:, :]
    p3 = x[s:, 0:s, :]
    p4 = x[s:, s:, :]

    if np.random.rand() < 0.5:
        return np.transpose(np.vstack([p1, p3]), (1,0,2))
    return np.transpose(np.vstack([p2, p4]), (1,0,2))

def generate_unmatch(x):
    s = x.shape[0] // 2
    p = []
    p.append(x[0:s, 0:s, :])
    p.append(x[0:s, s:, :])
    p.append(x[s:, 0:s, :])
    p.append(x[s:, s:, :])

    i = int(np.random.rand() * 4)
    if i > 3:
        i = 3
    if i == 0:
        j = 3
    elif i == 1:
        j = 2
    elif i == 2:
        j = 1
    else:
        j = 0

    if np.random.rand() < 0.5:
        return np.hstack([p[i], p[j]])
    return np.transpose(np.vstack([p[i], p[j]]), (1,0,2))

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, Xs, batch_size=256, puzzle_dim = 2, dim=(32,64,3)):
        'Initialization'
        self.batch_size = batch_size
        self.dim = dim
        self.puzzle_dim = puzzle_dim
        self.Xs = Xs
        self.on_epoch_end()
        model = network()

        model.load_weights('./models/weights.h5')

        self.model = Sequential()
        for layer in model.layers[:-2]:
            self.model.add(layer)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.Xs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate data
        X, y = self.__data_generation()

        return X, y

    def on_epoch_end(self):
        pass


    def __data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, ))

        n = np.random.randint(0, len(self.Xs) - 1, self.batch_size)
        j = 0
        for i in n:
            if np.random.rand() < 0.5:
                #matching
                if np.random.rand() < 0.5:
                    #vert
                    X[j] = generate_match_vert(self.Xs[i])
                    y[j] = 1
                else:
                    #horiz
                    X[j] = generate_match_horiz(self.Xs[i])
                    y[j] = 1
            else:
                #unmatching
                X[j] = generate_unmatch(self.Xs[i])
                y[j] = 0
            j += 1

        X = self.model.predict(X, batch_size=1024)

        return X, y
