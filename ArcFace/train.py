from PIL import Image
import os
import numpy as np
from tensorflow.keras.optimizers import Adam

from archs import resnet50_arcface


def main():
    imgs = []
    labels = []

    images = os.listdir("/Users/wantedly150/Downloads/Images/")
    for img in images:
        if not "jpg" in img: continue
        pil_x = Image.open("/Users/wantedly150/Downloads/Images/" + img)
        pil_x = pil_x.resize((112, 112))
        np_x = np.asarray(pil_x) / 255.0
        if np_x.shape != (112, 112, 3):
            continue
    
        label = [0] * 1000
        num = int(img.split("-")[1].split(".")[0])
        label[num] = 1
        imgs.append(np_x)
        labels.append(label)

    model = resnet50_arcface()
    model.compile(loss="categorical_crossentropy",
        optimizer=Adam(lr=0.01),
        metrics=["accuracy"]
    )
    model.summary()
    model.fit([imgs, labels], [labels],
        validation_data=([imgs, labels], [labels]),
        batch_size=50,
        epochs=5,
        verbose=1
    )

if __name__ == "__main__":
    main()