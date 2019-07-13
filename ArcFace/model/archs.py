import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Dropout, Flatten,
                                    Dense, Input, Softmax, Lambda, Layer)
from tensorflow.keras.backend import (l2_normalize, clip, epsilon, softmax)
from tensorflow.keras.regularizers import l2, get
from tensorflow.keras.models import load_model
import numpy as np
import os

class ArcFace(Layer):
    def __init__(self, n_classes=10, enhance=64.0, penalty=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = enhance
        self.m = penalty 
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
        x = l2_normalize(x, axis=1)
        W = l2_normalize(self.W, axis=0)
        logits = x @ W
        theta = tf.math.acos(clip(logits, -1.0 + epsilon(), 1.0 - epsilon()))
        target_logits = tf.math.cos(theta + self.m)
        logits = logits * (1 - y) + target_logits * y
        logits *= self.s
        out = softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)


def resnet50_arcface(args):
    n_classes = args.n_classes
    penalty = args.penalty
    enhance = args.enhance
    dropout_rate = args.dropout
    decay = args.decay

    resnet50 = tf.keras.applications.ResNet50(
        include_top=False, 
        input_shape=(100, 100, 3), 
        classes=n_classes
    )

    for layer in resnet50.layers:
        if "kernel_regularizer" in layer.__dict__:
            layer.kernel_regularizer = l2(decay)

    y = tf.keras.Input(shape=(n_classes,))

    x = resnet50.output
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = ArcFace(n_classes, enhance, penalty, l2(decay))([x, y])

    model = tf.keras.Model(inputs=[resnet50.input, y], outputs=x)
    if args.param_folder != "":
        path_params = os.path.dirname(os.path.abspath(__file__)) + \
            f"/../params/{args.param_folder}/params.hdf5"
        model.load_weights(path_params)

    return model
    
