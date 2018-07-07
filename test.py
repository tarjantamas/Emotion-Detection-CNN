from constants import REGULARIZED_MODEL_PATH, REGULARIZED_WEIGHTS_PATH
from keras.models import load_model
from data import test_generator

model = load_model(REGULARIZED_MODEL_PATH)
model.load_weights(REGULARIZED_WEIGHTS_PATH)
model.summary()
scores = model.evaluate_generator(test_generator, steps=1000)
print("Accuracy = ", scores[1])

