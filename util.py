import math
import dlib
import numpy as np
import cv2
import random


flip = np.array([-1, 1, 1], dtype=np.float)

def generate_image_set(image, center, size, fwd, down):
    angle = random.random() * math.pi / 2
    angleDeg = math.degrees(angle)
    scale = random.random() + 0.5
    x, y = center
    x0 = x - size // 2
    y0 = y - size // 2
    s = math.sin(angle)
    c = math.cos(angle)
    M = np.array([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1]
    ], dtype=np.float)
    padded = cv2.copyMakeBorder(image, size, size, size, size, cv2.BORDER_ISOLATED)
    center_crop = padded[y0 + size:y0 + 2 * size, x0 + size:x0 + 2 * size]
    am = cv2.getRotationMatrix2D(center, angleDeg, scale)
    am[0, 2] -= x0
    am[1, 2] -= y0
    yield [center_crop, [*fwd.flatten()], [*down.flatten()]]
    fwd = np.matmul(M, fwd)
    down = np.matmul(M, down)
    image = cv2.warpAffine(image, am, (size, size))
    yield [image, [*fwd.flatten()], [*down.flatten()]]
    image = cv2.flip(image, 1)
    yield [image, [*(fwd * flip).flatten()], [*(down * flip).flatten()]]


def crop_bb(image, bb):
    x, y, size = bb
    hsize = size // 2
    padded = cv2.copyMakeBorder(image, hsize, hsize, hsize, hsize, cv2.BORDER_ISOLATED)
    return padded[x:x + size, y:y + size]


def create_gen_from_file_listing(file_listing):
    while True:
        for _, select in file_listing.iterrows():
            print(select)
            img = cv2.imread(select['filename'], cv2.IMREAD_GRAYSCALE)
            fwd = np.array(select[['fx', 'fy', 'fz']], dtype=np.float)
            down = np.array(select[['dx', 'dy', 'dz']], dtype=np.float)
            center = tuple(select[['cx', 'cy']])
            size = select['size']
            for i, f, d in generate_image_set(img, center, size, fwd, down):
                i = np.reshape(cv2.resize(i, (224, 224)), (224, 224, 1))
                yield i, {'out_fwd': f, 'out_down': d}

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
    return cx, cy, size

