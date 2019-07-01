import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Dropout, Flatten,
                                    Dense, Input, Softmax, Lambda)
from tensorflow.keras.backend import l2_normalize, variable, clip, epsilon
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json, load_model
import numpy as np
from model.archs import ArcFace

if __name__ == "__main__":
    from PIL import Image

    num = 0
    base_url = "./lab-cardimage-match/params/20190701-154703/"
    model = load_model(
        base_url + "params.hdf5",
        custom_objects={"ArcFace" : ArcFace(m=0)}
    )
    
    #model.summary()

    img = Image.open(f"./sample-images/image-{num}/image-{num}.jpg")
    img = img.resize((220, 130))
    img = np.asarray(img).astype(np.float32)
    img = (img - 127.5) / 128.0


    label = np.zeros(10)
    #label[num] = 1
    pred = model.predict([[img], [label]])
    print(pred)
    print(np.argmax(pred))