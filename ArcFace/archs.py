import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Lambda, Softmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.backend import l2_normalize, variable, clip, epsilon
from tensorflow.keras.optimizers import Adam

import numpy as np
from prepare_data import generate_images

n_classes = 10
penalty = 0.5
trainfile = "/home/yudai/Documents/Python/lab-cardimage-match/lab-cardimage-match/sample-images"
validationfile = "/home/yudai/Documents/Python/lab-cardimage-match/lab-cardimage-match/validation-images"
#trainfile = "/home/yudai/Pictures/raw-img/train"
#validationfile = "/home/yudai/Pictures/raw-img/validation"

def vgg16_arcface(n_classes, penalty, decay):
    vgg16 = tf.keras.applications.VGG16(include_top=False, input_shape=(100, 100, 3), classes=n_classes)
    for layer in vgg16.layers:
        layer.trainable = False

    W = variable(np.random.randn(512, n_classes))
    W = l2_normalize(W, axis=0)
    y = tf.keras.Input(shape=(n_classes,))

    x = vgg16.output
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Lambda(lambda x: l2_normalize(x, axis=1))(x)
    x = Lambda(lambda x: x @ W)(x)

    theta = tf.acos(clip(x, -1.0 + epsilon(), 1.0 - epsilon()))
    target = tf.cos(theta + penalty)
    x = Lambda(lambda x: x[0]*(1-x[1]) + target*x[1])([x, y])

    x = Softmax()(x)

    model = tf.keras.Model([vgg16.input, y], x)
    model.compile(optimizer=Adam(0.01),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    return model


if __name__ == "__main__":
    train = generate_images(trainfile, 10)
    val = generate_images(validationfile, 10)

    model = vgg16_arcface(n_classes, penalty, 1e-4)
    history = model.fit_generator(train, steps_per_epoch=5, 
        epochs=100, 
        validation_steps=5, 
        validation_data=val
    )
