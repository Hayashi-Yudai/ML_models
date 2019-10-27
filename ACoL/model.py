import tensorflow as tf


class subbranch:
    def __call__(self, inputs):
        return self.sub_block(inputs)

    def sub_block(self, x):
        x = tf.keras.layers.Conv2D(
            filters=1024,
            kernel_size=3,
            padding="SAME",
            activation=tf.keras.activations.relu,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(
            filters=1024,
            kernel_size=3,
            padding="SAME",
            activation=tf.keras.activations.relu,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(
            filters=10,
            kernel_size=1,
            padding="SAME",
            activation=tf.keras.activations.relu,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        features = x
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Softmax()(x)

        return x, features


class Adversarial(tf.keras.layers.Layer):
    def __init__(self, batch_size, threshold):
        super(Adversarial, self).__init__(self)
        self.batch_size = batch_size
        self.threshold = threshold

    def call(self, inputs, **kwargs):
        vgg_end, interm, branchA_end = inputs  # (?, 7, 7, 512), (?, 7, 7, 10)
        max_idx = tf.argmax(branchA_end, axis=1)

        tmp = []

        # for bt in range(self.batch_size):
        for bt in range(self.batch_size):
            try:
                a = tf.reshape(interm[bt, :, :, max_idx[bt]], [7, 7, 1])
                each = tf.tile(a, [1, 1, 512])

                tmp.append(each)
            except:
                break

        tmp = tf.stack(tmp)
        tmp = tf.where(tmp > self.threshold, tmp, tmp * 0)

        adv = tf.subtract(vgg_end, tmp)

        return adv

    def compute_output_shape(self, input_shape):
        return (None, 7, 7, 512)


def ACoL(args):
    n_classes = args.n_classes
    batch_size = args.batch_size
    threshold = args.threshold

    vgg16 = tf.keras.applications.VGG16(
        include_top=False, input_shape=(224, 224, 3), classes=n_classes
    )

    for layer in vgg16.layers:
        layer.trainable = False
        if "kernel_regularizer" in layer.__dict__:
            layer.kernel_regularizer = tf.keras.regularizers.l2(0.001)

    x = vgg16.output  # (?, 7, 7, n_classes)

    # branch-A
    x, featuresA = subbranch()(x)

    # branch-B
    y = Adversarial(batch_size, threshold)([vgg16.output, featuresA, x])
    y, featuresB = subbranch()(y)
    output = tf.keras.layers.Lambda(lambda x: tf.add(x[0], x[1]))([x, y])

    return tf.keras.Model(inputs=vgg16.input, outputs=output)


if __name__ == "__main__":
    import numpy as np

    model = ACoL()
    print(model.predict(np.ones((1, 224, 224, 3))))
