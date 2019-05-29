import numpy as np
import tensorflow as tf
from tensorflow import keras

import util


def create_model():
    inputs = keras.Input(shape=(util.FEATURES,))
    hidden_1 = keras.layers.Dense(30, activation='relu', kernel_regularizer='l2')(inputs)
    hidden_2 = keras.layers.Dense(10, activation='relu', kernel_regularizer='l2')(hidden_1)
    predictions = keras.layers.Dense(2, activation='linear')(hidden_2)

    model = keras.Model(inputs=inputs, outputs=predictions)
    model.compile(
        optimizer=tf.train.AdamOptimizer(0.001), 
        loss='mean_squared_error',
        metrics=['mean_absolute_error', 'mean_squared_error'])
    return model
