import model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from typing import Tuple


class args:
    n_classes = 2
    l2 = 0.0
    weights = "./params/model.h5"
    img = "/home/yudai/Pictures/raw-img/train/cat/207.jpeg"


def prediction(img: np.ndarray) -> np.ndarray:
    unet = model.UNet(args)
    pred = unet.predict([img])

    return pred


def to_colormap(img: np.ndarray, original_shape: Tuple[int]) -> np.ndarray:
    pred = np.argmax(img[0], axis=2)
    ident = np.identity(3, dtype=np.uint8)
    pred = ident[pred] * 255
    pred = Image.fromarray(pred)
    pred = pred.resize(original_shape)

    return pred


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))

    img_original = Image.open(args.img).convert("RGB")
    original_shape = img_original.size

    img = img_original.resize((224, 224))
    img = np.asarray(img)
    img = img / 255.0

    pred = prediction(img.reshape(1, 224, 224, 3))
    pred = to_colormap(pred, original_shape)

    plt.imshow(img_original)
    plt.imshow(pred, alpha=0.3)
    plt.show()
