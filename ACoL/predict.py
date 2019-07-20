import tensorflow as tf
from PIL import Image
import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt

from model import ACoL
import train


class segmentation(tf.keras.layers.Layer):
    def __init__(self, batch_size):
        super(segmentation, self).__init__()
        self.batch_size = batch_size

    def call(self, inputs):
        x, y, flat = inputs
        idx = tf.argmax(flat, axis=1)

        res_x = []
        res_y = []
        for bt in range(self.batch_size):
            res_x.append(x[bt, :, :, idx[bt]])
            res_y.append(y[bt, :, :, idx[bt]])

        res_x = tf.stack(res_x)
        res_y = tf.stack(res_y)

        return tf.add(res_x, res_y)


def ACoL_predict(args):
    model = ACoL(args)
    model.load_weights(args.use_param)

    featureA = model.get_layer("batch_normalization_2").output
    featureB = model.get_layer("batch_normalization_5").output
    softmax = model.get_layer("softmax").output

    output_img = segmentation(args.batch_size)([featureA, featureB, softmax])

    return tf.keras.Model(inputs=model.input, outputs=output_img)


def predict(img, model):
    pil_img = Image.open(img).convert("RGB").resize((224, 224))
    np_imgs = (np.asarray(pil_img) - 127.5) / 128

    roi = model.predict(np_imgs.reshape((1, 224, 224, 3)))

    res = roi[0]
    space = int((16 * 7 - 2) / (7 - 1))
    x = [space * i for i in range(7)]
    y = [space * i for i in range(7)]
    x[-1] = 223
    y[-1] = 223
    f = interp2d(x, y, res)
    xx = [i for i in range(224)]
    yy = [i for i in range(224)]
    res = f(xx, yy)

    plt.figure()
    plt.imshow((np_imgs * 128 + 127.5).astype(np.int16))
    plt.imshow(res, cmap="seismic", alpha=0.3)
    plt.show()


if __name__ == "__main__":
    args = train.get_parser().parse_args()
    model = ACoL_predict(args)

    predict(
        "/home/yudai/Pictures/raw-img/validation/cow/OIP-0SwWnZTgIxQGLEKAqkSrdwHaFc.jpeg",
        model,
    )
