import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Dropout, Flatten,
                                    Dense, Input)

from metrics import ArcFace

def vgg16_arcface(n_class, m, decay):
    vgg16 = tf.keras.applications.VGG16(
        input_shape=[130, 220, 3], include_top=False
    )
    for layer in vgg16.layers:
        layer.trainable = False

    inputs = vgg16.input
    y = Input(shape=(n_class,))
    x = vgg16.output
    x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)
    x = Flatten()(x)
    x = Dense(512, kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(decay))(x)
    x = BatchNormalization()(x)

    output = ArcFace(n_class, m=m, regularizer=tf.keras.regularizers.l2(decay))([x, y])
    
    return tf.keras.Model([inputs, y], output)
