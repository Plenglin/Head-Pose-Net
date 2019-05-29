import re
import os
import csv


images = []


def add_image(path, yaw, pitch):
    images.append((path, yaw, pitch))


for person in os.listdir('data/pose2'):
    person_dir = 'data/pose2/' + person
    if not os.path.isdir(person_dir):
        continue
    for image in os.listdir(person_dir):
        match = re.match(r'[AB]_\d{2}_([+-]\d{2}).jpg', image)
        if not match:
            continue
        yaw = match.group(1)
        add_image(person_dir + '/' + image, float(yaw), 0.0)


for person in os.listdir('data/HeadPoseImageDatabase'):
    person_dir = 'data/HeadPoseImageDatabase/' + person
    if not os.path.isdir(person_dir):
        continue
    for image in os.listdir(person_dir):
        match = re.match(r'person\d{5}([+-]\d{2})([+-]\d{2}).jpg', image)
        if not match:
            continue
        yaw = match.group(2)
        pitch = match.group(1)
        add_image(person_dir + '/' + image, float(yaw), float(pitch))


with open('data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow('Path Yaw Pitch'.split())
    for row in images:
        writer.writerow(row)
