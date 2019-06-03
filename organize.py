import re
import os
import csv
import math
import numpy as np


images = []


def add_image(path, fwd, down):
    images.append((path, fwd, down))


for person in os.listdir('data/pose2'):
    person_dir = 'data/pose2/' + person
    if not os.path.isdir(person_dir):
        continue
    for image in os.listdir(person_dir):
        match = re.match(r'[AB]_\d{2}_([+-]\d{2}).jpg', image)
        if not match:
            continue
        yaw = math.radians(float(match.group(1)))
        add_image(person_dir + '/' + image, (math.sin(yaw), 0.0, math.cos(yaw)), (0.0, 1.0, 0.0))


for person in os.listdir('data/HeadPoseImageDatabase'):
    person_dir = 'data/HeadPoseImageDatabase/' + person
    if not os.path.isdir(person_dir):
        continue
    for image in os.listdir(person_dir):
        match = re.match(r'person\d{5}([+-]\d{2})([+-]\d{2}).jpg', image)
        if not match:
            continue
        yaw = float(match.group(2))
        pitch = float(match.group(1))

        cy = math.cos(yaw)
        sy = math.sin(yaw)
        cp = math.cos(pitch)
        sp = math.sin(pitch)

        fd = np.array([[1, 0], [0, 1], [0, 0]], dtype=np.float)
        fd = np.matmul([
                [1, 0, 0], 
                [0, cp, sp], 
                [0, -sp, cp]
        ], fd)
        f, d = np.matmul([
            [cy, 0, sy], 
            [0, 1, 0], 
            [-sy, 0, cy]
        ], fd).T

        add_image(person_dir + '/' + image, f, d)


with open('data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow('filename fx fy fz dx dy dz'.split())
    for p, f, d in images:
        #print(p, f, d)
        writer.writerow((p, *f, *d))
