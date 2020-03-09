import os
import random
import json

import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

from load import load_train_data
from generator import *
from model import denoising_network


if __name__ == '__main__':
    X = load_train_data(1)
    X_test = load_train_data(2)[0:10000]

    dim = (32,64,3)
    batch_size = 2**8
    puzzle_dim = 2

    train_params = {'dim': dim,
            'batch_size': batch_size,
            'puzzle_dim': puzzle_dim,
            'autoencoder': True,
            'noise': True,
            'noise_prob': 0.1}

    test_params = {'dim': dim,
            'batch_size': 16,
            'puzzle_dim': puzzle_dim,
            'autoencoder': True,
            'noise': True,
            'noise_prob': 0.1}

    generator_train = DataGenerator(X, **train_params)
    generator_test = DataGenerator(X_test, **test_params)

    model = denoising_network()

    if os.path.exists('./models/weights_denoise.h5'):
        model.load_weights('./models/weights_denoise.h5')

    for X_test, Y_test in generator_test:
        X_pred = model.predict(X_test)
        for x_t, x_p, y_t in zip(X_test, X_pred, Y_test):
            plt.subplot(3,1,1)
            plt.imshow(x_t)

            plt.subplot(3,1,2)
            plt.imshow(x_p)

            plt.subplot(3,1,3)
            plt.imshow(y_t)
            plt.show()

    exit()

    model.fit_generator(generator_train,
	validation_data=generator_test, steps_per_epoch=X.shape[0] // batch_size,
	epochs=5)

    model.save_weights('./models/weights_denoise.h5')
