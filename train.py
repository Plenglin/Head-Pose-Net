import csv

import cv2
import dlib
import numpy as np
import tensorflow as tf


class Prediction:
    def __init__(self):
        self.points = []
    
    #def distances(self, )

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

with open('data.csv', 'r') as file:
    data = list(csv.DictReader(file))
im_path = data[12]['Path']
img = cv2.imread(im_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)
shape = predictor(gray, rects[0])

for i in range(0, 68):
    pt = shape.part(i)
    cv2.circle(img, (pt.x, pt.y), 1, (0, 255, 0))

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
