from typing import Optional
import tensorflow as tf


class conv_set:
    def __init__(self, filters: int):
        self.filters = filters

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        y = tf.keras.layers.Conv2D(
            self.filters, kernel_size=3, padding="SAME", activation="relu"
        )(inputs)
        y = tf.keras.layers.Conv2D(
            self.filters, kernel_size=3, padding="SAME", activation="relu"
        )(y)
        y = tf.keras.layers.BatchNormalization()(y)
        return y


class upsampling:
    def __init__(self, filters: int, cut: Optional[int] = 0):
        self.filters = filters
        self.cut = cut

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        upconv = tf.keras.layers.Conv2DTranspose(
            self.filters, kernel_size=2, strides=2
        )(inputs[0])

        conv_crop = tf.keras.layers.Cropping2D(self.cut)(inputs[1])
        concat = tf.keras.layers.concatenate([conv_crop, upconv])

        return concat


def UNet(args: dict) -> tf.keras.Model:
    n_classes: int = args["n_classes"]
    decay: float = args["l2"]

    x = tf.keras.Input(shape=(224, 224, 3))

    # down sampling
    conv1 = conv_set(64)(x)
    max_pool1 = tf.keras.layers.MaxPool2D()(conv1)
    conv2 = conv_set(128)(max_pool1)
    max_pool2 = tf.keras.layers.MaxPool2D()(conv2)
    conv3 = conv_set(256)(max_pool2)
    max_pool3 = tf.keras.layers.MaxPool2D()(conv3)
    conv4 = conv_set(512)(max_pool3)
    max_pool4 = tf.keras.layers.MaxPool2D()(conv4)
    conv5 = conv_set(1024)(max_pool4)

    # up sampling
    concat1 = upsampling(512)([conv5, conv4])
    conv6 = conv_set(512)(concat1)
    concat2 = upsampling(256)([conv6, conv3])
    conv7 = conv_set(256)(concat2)
    concat3 = upsampling(128)([conv7, conv2])
    conv8 = conv_set(128)(concat3)
    concat4 = upsampling(64)([conv8, conv1])
    conv9 = conv_set(64)(concat4)

    output = tf.keras.layers.Conv2D(filters=n_classes, kernel_size=1)(conv9)
    output = tf.keras.layers.Softmax()(output)

    model = tf.keras.Model(inputs=x, outputs=output)
    for layer in model.layers:
        if "kernel_regularizer" in layer.__dict__:
            layer.kernel_regularizer = tf.keras.regularizers.l2(decay)

    if args["weights"] != "":
        model.load_weights(args["weights"])

    return model
