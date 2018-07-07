# Emotion detection using CNNs

## Running tensorbaord
`tensorboard --logdir .\logs\ --host=127.0.0.1`

## Generating the images
You can download this dataset [Here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
We are using the fer2013 dataset which is provided in csv format.
Each entry consists of a label and the pixels of the given image.
You can export the data into images using
`python export_images.py`

## Training
`python train.py`

## Evaluating the model using the test data
`python test.py`

## IP cam integration
You can download an ip cam app [here](https://play.google.com/store/apps/details?id=com.pas.webcam&hl=en).
Turn on your ip camera and run
`python cam.py`

You may have to modify the url variable in `cam.py`.
The default url is `'http://192.168.0.13:3256/photo.jpg'`