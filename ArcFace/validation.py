import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

from model.archs import ArcFace
from model.params_handler import parse_args


def generate_model(args):
    params = args.use_param_folder
    base_url = "./lab-cardimage-match/params/" + params
    model = load_model(
        base_url + "/params.hdf5",
        custom_objects={"ArcFace" : ArcFace}
    )

    model = tf.keras.Model(model.input, model.get_layer("dense").output)

    return model
    

def main(args, img):
    img = Image.open(img).convert("RGB")
    img = img.resize(220, 130)
    img = np.asarray(img)
    img = (img - 127.5) / 128
    label = np.zeros(args.n_classes)

    model = generate_model(args)

    return model.predict([[img], [label]])


if __name__ == "__main__":
    img = ""
    args = parse_args()
    embed = main(args, img)