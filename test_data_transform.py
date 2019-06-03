import cv2
import pandas as pd
import util
import numpy as np

file_listing = pd.read_csv('data.csv', header=0)
select = file_listing.iloc[0]

img = cv2.imread(select['filename'])
fwd = np.array(select[['fx', 'fy', 'fz']], dtype=np.float)
down = np.array(select[['dx', 'dy', 'dz']], dtype=np.float)

out = list(util.generate_image_set(img, fwd, down))
print(len(out))
for i, (im, f, d) in enumerate(out):
    print(f, d)
    cv2.imshow(str(i), im)
cv2.waitKey()
