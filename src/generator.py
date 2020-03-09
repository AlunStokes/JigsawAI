import cv2
import random

import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence

from utilities import *

def add_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.float32)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = np.array([0,0,0])
            elif rdn > thres:
                output[i][j] = np.array([1,1,1])
            else:
                output[i][j] = image[i][j]
    return output

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
    def __init__(self, Xs, batch_size=256, puzzle_dim = 2, dim=(32,64,3), autoencoder = False, noise = False, noise_prob = 0.2):
        'Initialization'
        self.batch_size = batch_size
        self.dim = dim
        self.puzzle_dim = puzzle_dim
        self.Xs = Xs
        self.autoencoder = autoencoder
        self.noise = noise
        self.noise_prob = noise_prob
        self.on_epoch_end()

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
        if self.autoencoder:
            Y = np.empty((self.batch_size, *self.dim))
        else:
            y = np.empty((self.batch_size, ))

        n = np.random.randint(0, len(self.Xs) - 1, self.batch_size)
        j = 0
        for i in n:
            if np.random.rand() < 0.5:
                #matching
                if np.random.rand() < 0.5:
                    #vert
                    X[j] = generate_match_vert(self.Xs[i])
                    if not self.autoencoder:
                        y[j] = 1
                else:
                    #horiz
                    X[j] = generate_match_horiz(self.Xs[i])
                    if not self.autoencoder:
                        y[j] = 1
            else:
                #unmatching
                X[j] = generate_unmatch(self.Xs[i])
                if not self.autoencoder:
                    y[j] = 0
            if self.autoencoder:
                Y[j] = X[j]
            if self.noise:
                X[j] = add_noise(X[j], self.noise_prob)
            j += 1
        if self.autoencoder:
            return X, Y
        else:
            return X, y
