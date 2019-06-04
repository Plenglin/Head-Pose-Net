import numpy as np
import tensorflow as tf
from tensorflow import keras

import util


def create_model():
    # Adapted AlexNet
    inputs = keras.Input(shape=(224, 224, 1), name='input')

    # 224
    conv_1 = keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu', name='conv_1a')(inputs)
    conv_1 = keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu', name='conv_1b')(conv_1)
    conv_1 = keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu', name='conv_1c')(conv_1)
    pool_1 = keras.layers.MaxPool2D((2, 2), name='pool_1')(conv_1)

    # 112
    conv_2 = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', name='conv_2a')(pool_1)
    conv_2 = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', name='conv_2b')(conv_2)
    conv_2 = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', name='conv_2c')(conv_2)
    pool_2 = keras.layers.MaxPool2D((2, 2), name='pool_2')(conv_2)

    # 56
    conv_3 = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv_3a')(pool_2)
    conv_3 = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv_3b')(conv_3)
    conv_3 = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv_3c')(conv_3)
    pool_3 = keras.layers.MaxPool2D((2, 2), name='pool_3')(conv_3)

    # 28
    conv_4 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv_4a')(pool_3)
    conv_4 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv_4b')(conv_4)
    pool_4 = keras.layers.MaxPool2D((2, 2), name='pool_4')(conv_4)

    # 14
    conv_5 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv_5a')(pool_4)
    conv_5 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv_5b')(conv_5)
    pool_5 = keras.layers.MaxPool2D((2, 2), name='pool_5')(conv_5)

    # 7
    conv_6 = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv_5a')(pool_5)
    conv_6 = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv_5b')(conv_6)
    pool_6 = keras.layers.MaxPool2D((2, 2), name='pool_6')(conv_6)

    # Hidden
    hidden_1 = keras.layers.Dense(20, activation='relu', kernel_regularizer='l2')(pool_6)
    drop_1 = keras.layers.Dropout(rate=0.5, name='drop_1')(hidden_1)
    hidden_2 = keras.layers.Dense(20, activation='relu', kernel_regularizer='l2')(drop_1)
    drop_2 = keras.layers.Dropout(rate=0.5, name='drop_2')(hidden_2)
    hidden_3 = keras.layers.Dense(20, activation='relu', kernel_regularizer='l2')(drop_2)
    drop_3 = keras.layers.Dropout(rate=0.5, name='drop_3')(hidden_3)
    outputs = keras.layers.Dense(6, activation='linear', name='outputs')(drop_3)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
