import csv
import util
import pandas as pd
import cv2


with open('data.csv', 'r') as file:
    data = list(csv.DictReader(file))

inputs = []
outputs = []

for i, row in enumerate(data):
    print(f'processing {i}')
    im_path = row['Path']
    img = cv2.imread(im_path)
    img = util.image_resize(img, height=400)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    prediction = util.do_prediction(gray)
    if prediction is None: 
        continue
    inputs.append(prediction.get_normalized_distances())
    outputs.append([row['Yaw'], row['Pitch']])

inputs = pd.DataFrame(inputs, columns=[f'len_{str(n).zfill(4)}' for n in range(68 * 67 // 2)])
outputs = pd.DataFrame(outputs, columns=['Yaw', 'Pitch'])
inputs.to_csv('ml_inputs.csv')
outputs.to_csv('ml_outputs.csv')
