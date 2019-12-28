import tensorflow as tf


class subbranch:
    def __init__(self, sign):
        self.sign = sign

    def __call__(self, inputs):
        return self.sub_block(inputs)

    def sub_block(self, x):
        x = tf.keras.layers.Conv2D(
            filters=1024,
            kernel_size=3,
            padding="SAME",
            activation=tf.keras.activations.relu,
            name=f"subbranch{self.sign}_conv1",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"subbranch{self.sign}_bn1")(x)
        x = tf.keras.layers.Conv2D(
            filters=1024,
            kernel_size=3,
            padding="SAME",
            activation=tf.keras.activations.relu,
            name=f"subbranch{self.sign}_conv2",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"subbranch{self.sign}_bn2")(x)
        x = tf.keras.layers.Conv2D(
            filters=10,
            kernel_size=1,
            padding="SAME",
            activation=tf.keras.activations.relu,
            name=f"subbranch{self.sign}_conv3",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"subbranch{self.sign}_bn3")(x)
        features = x
        x = tf.keras.layers.GlobalAveragePooling2D(name=f"subbranch{self.sign}_gap1")(x)
        x = tf.keras.layers.Softmax(name=f"subbranch{self.sign}_softmax1")(x)

        return x, features


class Adversarial(tf.keras.layers.Layer):
    def __init__(self, batch_size, threshold):
        super(Adversarial, self).__init__()
        self.batch_size = batch_size
        self.threshold = threshold

    def call(self, inputs):
        # vgg_end は VGG16 の出力, interm は branchA の GAP 直前 branchA_end は GAP, Softmax したあと
        vgg_end, interm, branchA_end = inputs  # (?, 7, 7, 512), (?, 7, 7, 10)
        max_idx = tf.argmax(branchA_end, axis=1)

        tmp = []

        for bt in range(self.batch_size):
            try:
                a = tf.reshape(interm[bt, :, :, max_idx[bt]], [7, 7, 1])
                each = tf.tile(a, [1, 1, 512])

                tmp.append(each)
            except Exception:
                break

        tmp = tf.stack(tmp)
        tmp = tf.where(tmp > self.threshold, tmp, tmp * 0)

        adv = tf.subtract(vgg_end, tmp)

        return adv
        return vgg_end

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
    x, featuresA = subbranch("A")(x)

    # branch-B
    y = Adversarial(batch_size, threshold)([vgg16.output, featuresA, x])
    y, featuresB = subbranch("B")(y)
    output = tf.keras.layers.Add()([x, y])

    return tf.keras.Model(inputs=vgg16.input, outputs=output)


if __name__ == "__main__":

    class p:
        n_classes = 10
        batch_size = 5
        threshold = 0.8

    m = ACoL(p())
    m.summary()
