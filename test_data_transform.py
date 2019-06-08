import cv2
import pandas as pd
import util
import numpy as np

file_listing = pd.read_csv('data.csv', header=0)
select = file_listing.iloc[3423]

for i, (im, l) in enumerate(util.create_gen_from_file_listing(file_listing)):
    print(l)
    cv2.imshow("image", im)
    cv2.waitKey()
