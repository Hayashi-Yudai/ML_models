import imgaug as iaa
import numpy as np
import os
from PIL import Image


def data_gen(train_data, seg_data, batch_size):
    inputs = np.array(os.listdir(train_data))
    batch = len(inputs) // batch_size
    identity = np.identity(2, dtype=np.int16)

    while True:
        shuffle = np.random.permutation(len(inputs))
        for b in np.array_split(inputs[shuffle], batch):
            imgs = []
            segs = []
            for img_file in b:
                img = Image.open(train_data + img_file).convert("RGB")
                img = img.resize((572, 572))
                img = np.asarray(img) / 255.0
                imgs.append(img)

                seg = Image.open(seg_data + img_file.split(".")[0] + "-seg.png")
                seg = seg.resize((388, 388))
                seg = np.asarray(seg)
                seg = identity[seg]
                segs.append(seg)

            yield np.array(imgs), np.array(segs)


if __name__ == "__main__":
    data = "./dataset/raw_images/"
    seg = "./dataset/segmented_images/"

    gen = data_gen(data, seg, 4)
    x, y = next(gen)
