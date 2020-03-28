import os
import random
import json

import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

from load import load_train_data
from generator import *
from model import network


if __name__ == '__main__':
    X = load_train_data('./processed/train')
    X_test = load_train_data('./processed/test')

    dim = (32,64,3)
    batch_size = 2**11
    puzzle_dim = 2

    params = {'dim': dim,
            'batch_size': batch_size,
            'puzzle_dim': puzzle_dim}

    generator_train = DataGenerator(X, **params)
    generator_test = DataGenerator(X_test, **params)

    for X,y in generator_train:
        i = 0
        while i < X.shape[0]:
            plt.imshow(X[i])
            plt.show()
            i += 1

    exit()

    model = network()

    if os.path.exists('./models/weights.h5'):
        model.load_weights('./models/weights.h5')

    model.fit_generator(generator_train,
	validation_data=generator_test, steps_per_epoch=X.shape[0] // batch_size,
	epochs=100)

    model.save_weights('./models/weights.h5')
