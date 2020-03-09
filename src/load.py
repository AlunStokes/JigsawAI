import pickle

import numpy as np

def load_train_data(num):
    train_file_name = './images/train64/{}'.format(num)

    with open(train_file_name, 'rb') as f:
        X = pickle.load(f)['data']

    X = X/np.float32(255)

    img_size = int((len(X[0]) // 3)**(1/2))
    img_size2 = img_size * img_size

    X = np.dstack((X[:, :img_size2], X[:, img_size2:2*img_size2], X[:, 2*img_size2:]))
    X = X.reshape((X.shape[0], img_size, img_size, 3)).transpose(0, 1, 2, 3)
    return X
