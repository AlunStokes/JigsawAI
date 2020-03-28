import os
import random
import json

import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

from load import load_train_data
from generator import *


if __name__ == '__main__':
    X = load_train_data('./processed/train')
    X_test = load_train_data('./processed/test')

    dim = (32,64,3)
    batch_size = 2**13
    puzzle_dim = 2

    params = {'dim': dim,
            'batch_size': batch_size,
            'puzzle_dim': puzzle_dim}

    generator_test = DataGenerator(X_test, **params)
    generator_train = DataGenerator(X, **params)

    for X, y in generator_train:
        break

    for X_test, y_test in generator_test:
        break

    print(X.shape)
    print(X_test.shape)

    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2] * X_test.shape[3])
    print(X.shape)
    print(X_test.shape)

    scores = []

    depth = 4
    n_estimator = 300

    epochs = 10

    if os.path.exists('./results/xgboost_res.json'):
        with open('./results/xgboost_res.json', 'r') as f:
            scores = json.load(f)

    print("Depth: {} \nNum Estimators: {}".format(depth, n_estimator))
    xgbc = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.1,
           max_delta_step=0, max_depth=depth, min_child_weight=1, missing=None,
           n_estimators=n_estimator, n_jobs=6, nthread=None,
           objective='binary:logistic', random_state=0, reg_alpha=0,
           reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
           subsample=1, verbosity=1)

    xgbc.fit(X, y)

    xgbc.save_model('./xgbc.model')

    y_pred = xgbc.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    print(cm)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy: {}".format(acc))
    print("")

    d = {
    'size_dataset': batch_size,
    'max_depth': depth,
    'n_estimators': n_estimator,
    'acc': acc
    }

    scores.append(d)

    with open('./results/xgboost_res.json', 'w') as f:
        json.dump(scores, f)
