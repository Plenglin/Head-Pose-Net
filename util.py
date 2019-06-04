import math
import dlib
import numpy as np
import cv2
import random



def generate_image_set(image, fwd, down, face_center):
    angle = random.random() * math.pi / 2
    angleDeg = math.degrees(angle)
    scale = random.random() + 0.5
    x, y = face_center
    s = math.sin(angle)
    c = math.cos(angle)
    M = np.array([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1]
    ], dtype=np.float)
    am = cv2.getRotationMatrix2D(face_center, angleDeg, scale)
    yield image[(x - 224)//2:(x+224)//2, (y - 224)//2:(y + 224)//2], fwd, down
    fwd = np.matmul(M, fwd)
    down = np.matmul(M, down)
    image = cv2.warpAffine(image, am, (224, 224))
    yield image, fwd, down
    flip = np.array([-1, 1, 1], dtype=np.float)
    image = cv2.flip(image, 1)
    yield image, fwd * flip, down * flip

# def create_gen_from_file_listing(file_listing, hairnet):

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

detector = dlib.get_frontal_face_detector()

def get_square_face_bb(img):
    rects = detector(img)
    if len(rects) == 0:
        return None
    rect = rects[0]
    x1 = rect.left()
    y1 = rect.top()
    x2 = rect.right()
    y2 = rect.bottom()
    #(x1, y1), (x2, y2) = rect
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    w = x2 - x1
    h = y2 - y1
    size = int(max(w, h) * 1.3)
    x0 = cx - size // 2
    y0 = cy - size // 2
    padded = cv2.copyMakeBorder(img, size, size, size, size, cv2.BORDER_ISOLATED)
    img = padded[y0 + size:y0 + 2 * size, x0 + size:x0 + 2 * size]
    return cx, cy, size

