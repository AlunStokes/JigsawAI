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

def inceptionV3(x, num):
    num = int(num)
    branch1 = Conv2D(num, (1,1), strides=(1,1), padding='same')(x)
    branch1 = BatchNormalization(axis=3)(branch1)
    branch1 = Activation('relu')(branch1)

    branch2 = Conv2D(num, (1,1), strides=(1,1), padding='same')(x)
    branch2 = BatchNormalization(axis=3)(branch2)
    branch2 = Activation('relu')(branch2)
    branch2 = Conv2D(num, (3,3), strides=(1,1), padding='same')(branch2)
    branch2 = BatchNormalization(axis=3)(branch2)
    branch2 = Activation('relu')(branch2)

    branch3 = Conv2D(num, (1,1), strides=(1,1), padding='same')(x)
    branch3 = BatchNormalization(axis=3)(branch3)
    branch3 = Activation('relu')(branch3)
    branch3 = Conv2D(num, (3,3), strides=(1,1), padding='same')(branch3)
    branch3 = BatchNormalization(axis=3)(branch3)
    branch3 = Activation('relu')(branch3)
    branch3 = Conv2D(num, (3,3), strides=(1,1), padding='same')(branch3)
    branch3 = BatchNormalization(axis=3)(branch3)
    branch3 = Activation('relu')(branch3)

    branch4 = Conv2D(num, (1,1), strides=(1,1), padding='same')(x)
    branch4 = BatchNormalization(axis=3)(branch4)
    branch4 = Activation('relu')(branch4)
    branch4 = Conv2D(num, (3,3), strides=(1,1), padding='same')(branch4)
    branch4 = BatchNormalization(axis=3)(branch4)
    branch4 = Activation('relu')(branch4)
    branch4 = Conv2D(num, (3,3), strides=(1,1), padding='same')(branch4)
    branch4 = BatchNormalization(axis=3)(branch4)
    branch4 = Activation('relu')(branch4)
    branch4 = Conv2D(num, (3,3), strides=(1,1), padding='same')(branch4)
    branch4 = BatchNormalization(axis=3)(branch4)
    branch4 = Activation('relu')(branch4)

    branch5 = AveragePooling2D((3,3), strides=(1,1), padding='same')(x)
    branch5 = Conv2D(num, (3,3), strides=(1,1), padding='same')(branch5)
    res = concatenate([branch1,branch2,branch3,branch4,branch5], axis=3)
    return res

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


def step(d, x, n, concat_inputs):
    convs_down = []
    pools = []
    ups = []
    convs_up = []
    concats = []

    convs_down.append(inceptionV3(x, n))
    pools.append(MaxPooling2D(pool_size=(2,2))(convs_down[-1]))

    i = 0
    while i < d - 1:
        convs_down.append(inceptionV3(pools[-1], 2**(i + 1)*n))
        pools.append(MaxPooling2D(pool_size=(2,2))(convs_down[-1]))
        i += 1
    convs_down.append(inceptionV3(pools[-1], 2**(i + 1)*n))
    i -= 1

    ups.append(Conv2DTranspose(filters=4*2**(i + 1)*n, kernel_size=(2,2), strides=(2,2), data_format='channels_last')(convs_down[-1]))
    c_arr = [ups[0], convs_down[-2]]
    for input in concat_inputs:
        c_arr.append(input[-1])
    concats.append(concatenate(c_arr, axis=3))
    convs_up.append(inceptionV3(concats[0], 2**i*n))

    j = 0
    while j < d - 1:
        ups.append(Conv2DTranspose(filters=8*2**i*n, kernel_size=(2,2), strides=(2,2), data_format='channels_last')(convs_up[-1]))
        c_arr = [ups[j + 1], convs_down[-(3 + j)]]
        for input in concat_inputs:
            c_arr.append(input[-(1 + j + 1)])
        concats.append(concatenate(c_arr, axis=3))
        convs_up.append(inceptionV3(concats[-1], 2**i*n))
        j += 1

    return convs_up[-1], convs_down[:-1]

def step_net(n ,length, depth):
    inputs = Input((64,64,3))
    steps = []

    concat_inputs = []

    x, c = step(depth, inputs, n, [])
    concat_inputs.append(c)
    steps.append(x)
    i = 0
    while i < length - 1:
        x, c = step(depth, x, n, concat_inputs)
        concat_inputs.append(c)
        steps.append(x)
        i += 1

    out = inceptionV3(steps[-1], n)
    out = MaxPooling2D(pool_size=(2,2))(out)

    out = inceptionV3(out, n)
    out = MaxPooling2D(pool_size=(2,2))(out)

    out = inceptionV3(out, n)
    out = MaxPooling2D(pool_size=(2,2))(out)

    out = inceptionV3(out, n)
    out = MaxPooling2D(pool_size=(2,2))(out)

    out = inceptionV3(out, n)
    out = MaxPooling2D(pool_size=(2,2))(out)

    out = Conv2D(4, (1,1))(out)
    out = Activation('sigmoid')(out)

    model = Model(inputs=inputs, outputs=out)
    #model.compile(optimizer=Adam(lr=1e-4), loss=create_weighted_binary_crossentropy(0.23, 1 - 0.23), metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=1e-4), loss=combined_mse_crossent, metrics=['mse'])
    model.summary()
    return model
