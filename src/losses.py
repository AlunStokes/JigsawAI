import tensorflow.keras.backend as K
from tensorflow.keras.losses import mse, binary_crossentropy

def combined_mse_crossent(y_true, y_pred):
    return mse(y_true, y_pred) + binary_crossentropy(y_true, y_pred)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(K.pow(y_true_f, 2)) + K.sum(K.pow(y_pred_f, 2)) + 1e-8)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
