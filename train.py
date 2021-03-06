import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

import model
import util
import datetime


LOG_DIR = "./logs/" + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
EPOCHS = 1000
STEPS_PER_EPOCH = 100
BATCH_SIZE = 30

file_listing = pd.read_csv("data.csv")

gen = lambda: util.create_gen_from_file_listing(file_listing)
dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32), ((224, 224, 1), (6,)))

def dot_error(y_true, y_pred):
    with tf.name_scope('dot_error'):
        dot = tf.reduce_sum(y_true * y_pred, 1)
        return dot - 1

def square_dot_error(y_true, y_pred):
    with tf.name_scope('square_dot_error'):
        return tf.square(dot_error(y_true, y_pred))

with tf.Session() as sess:
    tf.keras.backend.set_session(sess)
    with tf.name_scope('posenet'):
        posenet = model.create_model()
    iterator = (dataset
        .batch(BATCH_SIZE)
        .prefetch(8)
        .repeat()
        .make_one_shot_iterator())
    images, labels = iterator.get_next()

    posenet.compile(
        optimizer=tf.train.AdamOptimizer(0.001),
        loss={'out_fwd': square_dot_error, 'out_down': square_dot_error},
        loss_weights={"out_fwd": 1.0, "out_down": 1.0},
        #metrics=['mean_squared_error'],   
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=50)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        "training/cp-{epoch:04d}.ckpt", save_weights_only=True, verbose=1, period=10
    )
    tboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=LOG_DIR, batch_size=32, write_graph=True, update_freq="epoch"
    )

    posenet.fit(
        images,
        labels,
        epochs=EPOCHS,
        steps_per_epoch=100,
        callbacks=[early_stop, cp_callback, tboard_callback],
    )

    try:
        os.makedirs("./saved_models")
    except FileExistsError:
        pass
    tf.contrib.saved_model.save_keras_model(posenet, "./saved_models")
