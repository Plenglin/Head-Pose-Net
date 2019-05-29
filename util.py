import math
import dlib
import numpy as np
import cv2
from scipy.stats import zscore


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

class Prediction:
    def __init__(self):
        self.points = [None] * 68
    
    def get_distances(self):
        out = []
        for i in range(68):
            x1, y1 = self.points[i]
            for j in range(i + 1, 68):
                x2, y2 = self.points[j]
                dx = x1 - x2
                dy = y1 - y2
                out.append(math.sqrt(dx * dx + dy * dy))
        return out
    
    def get_normalized_distances(self):
        return zscore(self.get_distances())


def do_prediction(gray):
    rects = detector(gray, 1)
    shape = predictor(gray, rects[0])

    out = Prediction()
    for i in range(0, 68):
        pt = shape.part(i)
        out.points[i] = (pt.x, pt.y)
    return out

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized