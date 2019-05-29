import csv
import util
import pandas as pd
import cv2

import multiprocessing as mp


with open('data.csv', 'r') as file:
    data = list(csv.DictReader(file))

inputs = []
outputs = []


def worker_thread(row):
    im_path = row['Path']
    img = cv2.imread(im_path)
    img = util.image_resize(img, height=400)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    prediction = util.do_prediction(gray)
    if prediction is None: 
        return None
    return prediction.get_normalized_distances(), [row['Yaw'], row['Pitch']]

with mp.Pool(mp.cpu_count * 2) as pool:
    processed = pool.map(worker_thread, data)

inputs, outputs = zip(*processed)

inputs = pd.DataFrame(inputs, columns=[f'len_{str(n).zfill(4)}' for n in range(util.FEATURES)])
outputs = pd.DataFrame(outputs, columns=['Yaw', 'Pitch'])
inputs.to_csv('ml_inputs.csv', float_format='%.6f')
outputs.to_csv('ml_outputs.csv')
