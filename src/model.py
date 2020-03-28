import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Flatten, UpSampling2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Conv2DTranspose, BatchNormalization, Dropout, Input, Activation, concatenate

from metrics import *
from losses import *

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default

def network():
    inputs = Input((32,64,3))

    out = Conv2D(8, (3,3), activation='relu', padding='same')(inputs)
    out = MaxPooling2D((2,2))(out)

    out = Conv2D(16, (3,3), activation='relu', padding='same')(out)
    out = MaxPooling2D((2,2))(out)

    out = Conv2D(32, (3,3), activation='relu', padding='same')(out)
    out = MaxPooling2D((2,2))(out)

    out = Flatten()(out)

    out = Dense(128, activation='relu')(out)
    out = Dropout(0.3)(out)

    out = Dense(64, activation='relu')(out)
    out = Dropout(0.3)(out)

    out = Dense(32, activation='relu')(out)
    out = Dropout(0.3)(out)

    out = Dense(1, activation='sigmoid')(out)

    model = Model(inputs=inputs, outputs=out)
    #model.compile(optimizer=Adam(lr=1e-4), loss=create_weighted_binary_crossentropy(0.23, 1 - 0.23), metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def denoising_network():
    inputs = Input((32,64,3))

    out = Conv2D(128, (3,3), activation='relu', padding='same')(inputs)
    out = BatchNormalization()(out)
    out = Conv2D(128, (3,3), activation='relu', padding='same')(inputs)
    out = BatchNormalization()(out)
    out = MaxPooling2D((2,2))(out)

    out = Conv2D(64, (3,3), activation='relu', padding='same')(out)
    out = BatchNormalization()(out)
    out = Conv2D(64, (3,3), activation='relu', padding='same')(out)
    out = BatchNormalization()(out)
    out = MaxPooling2D((2,2))(out)

    out = Conv2D(32, (3,3), activation='relu', padding='same')(out)
    out = BatchNormalization()(out)
    out = Conv2D(32, (3,3), activation='relu', padding='same')(out)
    out = BatchNormalization()(out)
    out = MaxPooling2D((2,2))(out)

    out = Conv2DTranspose(8, (2,2), strides=(2,2), activation='relu')(out)
    out = Conv2D(32, (3,3), activation='relu', padding='same')(out)
    out = BatchNormalization()(out)
    out = Conv2D(32, (3,3), activation='relu', padding='same')(out)
    out = BatchNormalization()(out)

    out = Conv2DTranspose(16, (2,2), strides=(2,2), activation='relu')(out)
    out = Conv2D(64, (3,3), activation='relu', padding='same')(out)
    out = BatchNormalization()(out)
    out = Conv2D(64, (3,3), activation='relu', padding='same')(out)
    out = BatchNormalization()(out)

    out = Conv2DTranspose(32, (2,2), strides=(2,2), activation='relu')(out)
    out = Conv2D(128, (3,3), activation='relu', padding='same')(out)
    out = BatchNormalization()(out)
    out = Conv2D(128, (3,3), activation='relu', padding='same')(out)
    out = BatchNormalization()(out)

    out = Conv2D(3, (1, 1), activation='sigmoid', padding='same')(out)

    model = Model(inputs=inputs, outputs=out)
    #model.compile(optimizer=Adam(lr=1e-4), loss=create_weighted_binary_crossentropy(0.23, 1 - 0.23), metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['mse'])
    model.summary()
    return model
