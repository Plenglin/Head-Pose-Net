import numpy as np
import tensorflow as tf
from tensorflow import keras

import util


def create_model():
    # Adapted AlexNet
    inputs = keras.Input(shape=(224, 224, 1), name='input')

    conv_1 = keras.layers.Conv2D(96, (11, 11), (4, 4), padding='valid', activation='relu', name='conv_1')(inputs)
    pool_1 = keras.layers.MaxPool2D((3, 3), (2, 2), name='pool_1')(conv_1)

    conv_2 = keras.layers.Conv2D(256, (5, 5), (1, 1), padding='valid', activation='relu', name='conv_2')(pool_1)
    pool_2 = keras.layers.MaxPool2D((2, 2), (2, 2), name='pool_2')(conv_2)

    conv_3 = keras.layers.Conv2D(384, (3, 3), (1, 1), padding='valid', activation='relu', name='conv_3')(pool_2)
    conv_4 = keras.layers.Conv2D(384, (3, 3), (1, 1), padding='valid', activation='relu', name='conv_4')(conv_3)
    conv_5 = keras.layers.Conv2D(256, (3, 3), (1, 1), padding='valid', activation='relu', name='conv_5')(conv_4)
    pool_5 = keras.layers.MaxPool2D((3, 3), (2, 2), name='pool_5')(conv_5)

    flatten = keras.layers.Flatten(name='flatten')(pool_5)
    batch_norm = keras.layers.BatchNormalization(name='batch_norm')(flatten)

    # Hidden
    hidden_1 = keras.layers.Dense(4096, activation='relu', kernel_regularizer='l2', name='hidden_1')(batch_norm)
    drop_1 = keras.layers.Dropout(rate=0.5, name='drop_1')(hidden_1)
    hidden_2 = keras.layers.Dense(4096, activation='relu', kernel_regularizer='l2', name='hidden_2')(drop_1)
    drop_2 = keras.layers.Dropout(rate=0.5, name='drop_2')(hidden_2)
    hidden_3 = keras.layers.Dense(4096, activation='relu', kernel_regularizer='l2', name='hidden_3')(drop_2)
    drop_3 = keras.layers.Dropout(rate=0.5, name='drop_3')(hidden_3)
    outputs = keras.layers.Dense(6, activation='linear', name='output')(drop_3)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
