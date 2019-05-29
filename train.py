import csv

import cv2
import dlib
import numpy as np
import tensorflow as tf

import util


class Prediction:
    def __init__(self):
        self.points = []
    
    #def distances(self, )

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

with open('data.csv', 'r') as file:
    data = list(csv.DictReader(file))

im_path = data[12]['Path']
print(f'importing {im_path}')
img = cv2.imread(im_path)
img = util.image_resize(img, height=400)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('img',img)
cv2.waitKey(0)

print('prediction')
prediction = util.do_prediction(gray)
print('drawing')
for pt in prediction.points:
    cv2.circle(img, pt, 1, (0, 255, 0))

print(prediction.get_normalized_distances())

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
