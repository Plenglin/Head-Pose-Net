import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np

import time

import util
import math


def draw_axis(img, o, l, f, d):
    points = (np.array([f, d, np.cross(f, d)]) * l + o)
    points = points.astype(np.int)
    o = o.astype(np.int)
    proj_or = o[0], o[1]
    print(points)
    img = cv2.line(img, proj_or, (points[0, 0], points[0, 1]), (255,0,0), 3)
    img = cv2.line(img, proj_or, (points[1, 0], points[1, 1]), (0,255,0), 3)
    img = cv2.line(img, proj_or, (points[2, 0], points[2, 1]), (0,0,255), 3)
    return img

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
graph = tf.Graph()
with tf.Session(graph=graph, config=config) as sess:
    tf.keras.backend.set_session(sess)
    hairnet_def = tf.contrib.saved_model.load_keras_model("./saved_models/1559782143")

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

            (xf, yf, zf, xd, yd, zd), = sess.run(output_layer, feed_dict={"input:0": [ml_in]})
            f = np.array([xf, yf, zf])
            d = np.array([xd, yd, zd])
            draw_axis(frame, np.array([100., 100., 0.], dtype=np.float), 100, f, d)
            print(f"FPS: {1 / (time.time() - start)}, {f} {d} dp={np.dot(f, d)} p={math.degrees(math.atan2(yf, zf))} y={math.degrees(math.atan2(zf, xf))}")
        cv2.imshow('img', frame)
