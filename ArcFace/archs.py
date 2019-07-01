import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Dropout, Flatten,
                                    Dense, Input, Softmax, Lambda, Layer)
from tensorflow.keras.backend import l2_normalize, variable, clip, epsilon
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2, get
import numpy as np

class ArcFace(Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = get(regularizer)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(512, self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        x = tf.nn.l2_normalize(x, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)
        logits = x @ W
        theta = tf.acos(clip(logits, -1.0 + epsilon(), 1.0 - epsilon()))
        target_logits = tf.cos(theta + self.m)
        logits = logits * (1 - y) + target_logits * y
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)


def vgg16_arcface(n_classes, penalty, decay, fine_tune=None):
    vgg16 = tf.keras.applications.VGG16(
        include_top=False, 
        input_shape=(130, 220, 3), 
        classes=n_classes
    )
    if fine_tune is None:
        for layer in vgg16.layers:
            layer.trainable = False

    y = tf.keras.Input(shape=(n_classes,))

    x = vgg16.output
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = ArcFace(10, regularizer=l2(decay))([x, y])

    return tf.keras.Model(inputs=[vgg16.input, y], outputs=x)