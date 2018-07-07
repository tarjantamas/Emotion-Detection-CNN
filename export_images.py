import json
import numpy as np
from PIL import Image
import os, errno

try:
    os.makedirs('images/train')
    os.makedirs('images/validation')
    os.makedirs('images/test')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

with open("fer2013.csv", "r") as infile:
    trainIndex = 0
    validationIndex = 0
    testIndex = 0
    for line in infile:
        tokens = line.split(",")
        if tokens[0] == "emotion":
            continue
        emotion = tokens[0]
        image = tokens[1].strip('"')
        usage = tokens[2].strip()
        data = {
            "emotion": int(emotion),
            "image": [int(x.strip().strip('"')) for x in image.split(' ')]
        }
        image = np.reshape(data['image'], (48, 48))
        img = Image.fromarray(np.uint8(image), 'L')
        if usage == 'Training':
            img.save('./images/train/' + str(data['emotion']) + '/' + str(trainIndex) + '.png', 'png')
            trainIndex += 1
        elif usage == 'PublicTest':
            img.save('./images/validation/' + str(data['emotion']) + '/' + str(validationIndex) + '.png', 'png')
            validationIndex += 1
        elif usage == 'PrivateTest':
            img.save('./images/test/' + str(data['emotion']) + '/' + str(testIndex) + '.png', 'png')
            testIndex += 1
