import cv2
import pandas as pd
import util
import numpy as np

file_listing = pd.read_csv('data.csv', header=0)
select = file_listing.iloc[3423]

img = cv2.imread(select['filename'])
fwd = np.array(select[['fx', 'fy', 'fz']], dtype=np.float)
down = np.array(select[['dx', 'dy', 'dz']], dtype=np.float)
center = tuple(select[['cx', 'cy']])
size = select['size']

out = list(util.generate_image_set(img, center, size, fwd, down))
print(len(out))
for i, (im, f, d) in enumerate(out):
    print(f, d)
    cv2.imshow(str(i), im)
cv2.waitKey()
