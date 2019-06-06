import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np

import time

import util



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
graph = tf.Graph()
with tf.Session(graph=graph, config=config) as sess:
    tf.keras.backend.set_session(sess)
    hairnet_def = tf.contrib.saved_model.load_keras_model("./saved_models/1559611322")

    input_layer = graph.get_tensor_by_name("input:0")
    output_layer = graph.get_tensor_by_name("output/BiasAdd:0")

    cam = cv2.VideoCapture(0)
    while cv2.waitKey(1) & 0xFF != ord('q'):
        ret, frame = cam.read()
        start = time.time()
        bb = util.get_square_face_bb(frame)
        if bb is not None:
            cropped = util.crop_bb(frame, bb)
            resized = cv2.resize(cropped, (224, 224))
            ml_in = np.reshape(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), (224, 224, 1))

            result = sess.run(output_layer, feed_dict={"input:0": [ml_in]})
            print(f"FPS: {1 / (time.time() - start)}, {result}")
        cv2.imshow('img', frame)
