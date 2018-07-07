from constants import REGULARIZED_MODEL_PATH, REGULARIZED_WEIGHTS_PATH, MODEL_ROOT
from model import buildModel, buildModelMoreDenses
from data import train_generator, validation_generator
from callbacks import modelCheckpoint, tensorBoard
import os, errno

try:
    os.makedirs(MODEL_ROOT)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

model = buildModelMoreDenses()
model.summary()
model.save(REGULARIZED_MODEL_PATH)
model.load_weights(REGULARIZED_WEIGHTS_PATH)


model.fit_generator(
    train_generator,
    steps_per_epoch=50,
    epochs=200,
    validation_data=validation_generator,
    validation_steps=50,
    callbacks=[modelCheckpoint, tensorBoard])


model.save_weights(REGULARIZED_WEIGHTS_PATH)
