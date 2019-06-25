import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Dropout, Flatten,
                                    Dense, Input)

from metrics import ArcFace

weight_decay = 1e-4

def resnet50_arcface():
    resnet50 = tf.keras.applications.ResNet50(
        input_shape=[112, 112, 3], include_top=False
    )
    for layer in resnet50.layers:
        layer.trainable = False

    inputs = resnet50.input
    y = Input(shape=(1000,))
    x = resnet50.output
    x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)
    x = Flatten()(x)
    x = Dense(512, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)

    output = ArcFace(1000, regularizer=tf.keras.regularizers.l2(weight_decay))([x, y])

    return tf.keras.Model([inputs, y], output)


if __name__ ==  "__main__":
    model = resnet50_arcface()