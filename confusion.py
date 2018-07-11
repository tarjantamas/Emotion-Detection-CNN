from constants import REGULARIZED_MODEL_PATH, REGULARIZED_WEIGHTS_PATH
from keras.models import load_model
from data import test_generator
import tensorflow as tf
import numpy as np

model = load_model(REGULARIZED_MODEL_PATH)
model.load_weights(REGULARIZED_WEIGHTS_PATH)
model.summary()

batch_count = 200

labels, predictions = [], []

for i in range(batch_count):
    for x, y in test_generator:
        pred = model.predict_classes(x)
        y = np.argmax(y, axis=1)

        for y1 in y:
            labels.append(y1)

        for y1 in pred:
            predictions.append(y1)
            
        break



matrix = tf.confusion_matrix(labels, predictions)
a, b = tf.metrics.mean_per_class_accuracy(tf.convert_to_tensor(labels), tf.convert_to_tensor(predictions), 7)

sess = tf.Session()
sess.as_default()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
print(sess.run(matrix))
print(sess.run(a))
print(sess.run(b))
