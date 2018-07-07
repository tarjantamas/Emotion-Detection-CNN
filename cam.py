from PIL import Image
import requests
import numpy as np
from keras.models import load_model
from constants import REGULARIZED_MODEL_PATH, REGULARIZED_WEIGHTS_PATH
import time

emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear",
                3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

size = 48, 48
url = 'http://192.168.0.13:3256/photo.jpg'

while True:
    input('Take picture')
    
    image = Image.open(requests.get(url, stream=True).raw)
    image = image.convert('L')
    image = image.resize(size, Image.ANTIALIAS)

    model = load_model(REGULARIZED_MODEL_PATH)
    model.load_weights(REGULARIZED_WEIGHTS_PATH)
    imageArray = np.array(image) * 1./255
    imageArray = np.expand_dims(imageArray, -1)
    imageArray = np.expand_dims(imageArray, 0)
    print(imageArray.shape)
    result = model.predict(imageArray)
    print(emotion_dict[int(np.argmax(result))])

    
