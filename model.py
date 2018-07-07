from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Flatten, Activation, MaxPooling2D, Dense, Dropout
from keras.initializers import VarianceScaling
from keras import regularizers
from keras.optimizers import Adam

def buildModel():
    model = Sequential()

    model.add(Conv2D(
        input_shape=(48, 48, 1),
        filters=64,
        kernel_size=(3, 3),
        kernel_initializer=VarianceScaling(),
        activation='relu'
    ))

    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        kernel_initializer=VarianceScaling(),
        use_bias=False
    ))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ))

    model.add(Conv2D(
        filters=512,
        kernel_size=(3, 3),
        kernel_initializer=VarianceScaling(),
        use_bias=False
    ))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ))
    
    model.add(Conv2D(
        filters=512,
        kernel_size=(3, 3),
        kernel_initializer=VarianceScaling(),
        kernel_regularizer=regularizers.l2(0.001),
        use_bias=False
    ))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ))

    model.add(Flatten())

    model.add(Dense(
        units=512,
        kernel_initializer=VarianceScaling()
    ))
    model.add(Activation('relu'))

    model.add(Dense(
        units=7,
        kernel_initializer=VarianceScaling(),
        activation='softmax'
    ))

    model.compile(
        loss='categorical_crossentropy',
        optimizer="adam",
        metrics=['accuracy']
    )

    return model


def buildModelMoreDenses():
    model = Sequential()

    model.add(Conv2D(
        input_shape=(48, 48, 1),
        filters=64,
        kernel_size=(3, 3),
        kernel_initializer=VarianceScaling(),
        activation='relu',
        trainable=False
    ))
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ))

    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        kernel_initializer=VarianceScaling(),
        use_bias=False,
        trainable=False
    ))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ))

    model.add(Conv2D(
        filters=512,
        kernel_size=(3, 3),
        kernel_initializer=VarianceScaling(),
        use_bias=False,
        trainable=False
    ))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ))

    model.add(Conv2D(
        filters=512,
        kernel_size=(3, 3),
        kernel_initializer=VarianceScaling(),
        kernel_regularizer=regularizers.l2(0.001),
        use_bias=False,
        trainable=False
    ))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ))

    model.add(Flatten())

    model.add(Dense(
        units=512,
        kernel_initializer=VarianceScaling(),
        activation='relu'
    ))

    model.add(Dense(
        units=256,
        kernel_initializer=VarianceScaling(),
        activation='relu'
    ))

    model.add(Dense(
        units=128,
        kernel_initializer=VarianceScaling(),
        activation='relu'
    ))

    model.add(Dense(
        units=64,
        kernel_initializer=VarianceScaling(),
        activation='relu'
    ))

    model.add(Dense(
        units=32,
        kernel_initializer=VarianceScaling(),
        activation='relu'
    ))

    model.add(Dense(
        units=16,
        kernel_initializer=VarianceScaling(),
        activation='relu'
    ))

    model.add(Dense(
        units=7,
        kernel_initializer=VarianceScaling(),
        activation='softmax'
    ))

    model.compile(
        loss='categorical_crossentropy',
        optimizer="adam",
        metrics=['accuracy']
    )

    return model
