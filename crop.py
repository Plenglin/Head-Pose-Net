import cv2
import dlib
import pandas as pd


detector = dlib.get_frontal_face_detector()
listing = pd.read_csv('data.csv', header=0)
i = 0
for _, row in listing.iterrows():
    img = cv2.imread(row['filename'])
    rects = detector(img)
    if len(rects) == 0:
        continue
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
    i += 1
    filename = 'data/cropped/img_{:04d}.jpg'.format(i)
    img = cv2.resize(img, (224, 224))
    cv2.imwrite(filename, img)
    row['filename'] = filename

listing.to_csv('cropped.csv', index=False)
