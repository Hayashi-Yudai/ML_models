import tensorflow as tf


class conv_set:
    def __init__(self, filters):
        self.filters = filters

    def __call__(self, inputs):
        y = tf.keras.layers.Conv2D(self.filters, kernel_size=3, activation="relu")(
            inputs
        )
        y = tf.keras.layers.Conv2D(self.filters, kernel_size=3, activation="relu")(y)
        y = tf.keras.layers.BatchNormalization()(y)
        return y


class updampling:
    def __init__(self, filters, cut):
        self.filters = filters
        self.cut = cut

    def __call__(self, inputs):
        upconv = tf.keras.layers.Conv2DTranspose(
            self.filters, kernel_size=2, strides=2
        )(inputs[0])

        conv_crop = tf.keras.layers.Cropping2D(self.cut)(inputs[1])
        concat = tf.keras.layers.concatenate([conv_crop, upconv])

        return concat


def UNet(args):
    n_classes = args.n_classes
    decay = args.l2

    x = tf.keras.Input(shape=(572, 572, 3))

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
    concat1 = updampling(512, 4)([conv5, conv4])
    conv6 = conv_set(512)(concat1)
    concat2 = updampling(256, 16)([conv6, conv3])
    conv7 = conv_set(256)(concat2)
    concat3 = updampling(128, 40)([conv7, conv2])
    conv8 = conv_set(128)(concat3)
    concat4 = updampling(64, 88)([conv8, conv1])
    conv9 = conv_set(64)(concat4)

    output = tf.keras.layers.Conv2D(filters=n_classes, kernel_size=1)(conv9)

    model = tf.keras.Model(inputs=x, outputs=output)
    for layer in model.layers:
        if "kernel_regularizer" in layer.__dict__:
            layer.kernel_regularizer = tf.keras.regularizers.l2(decay)

    return model


if __name__ == "__main__":
    unet = UNet()
    print(unet.summary())
