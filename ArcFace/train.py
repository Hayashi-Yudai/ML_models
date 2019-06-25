from PIL import Image
import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import tensorflow as tf

from archs import resnet50_arcface
from prepare_data import make_dataset


def main():
    imgs, labels = make_dataset(10)

    shuffle_idx = np.random.permutation(500)
    imgs = np.array(imgs)[shuffle_idx]
    labels = np.array(labels)[shuffle_idx]

    X, X_test = imgs[:-200], imgs[-200:]
    y, y_test = labels[:-200], labels[-200:]

    model = resnet50_arcface(10)
    model.compile(loss="categorical_crossentropy",
        optimizer=Adam(lr=0.1),
        metrics=["accuracy"]
    )
    model.summary()
    model.fit([X, y], y,
        validation_data=([X_test, y_test], y_test),
        batch_size=50,
        epochs=500,
        verbose=1
    )

if __name__ == "__main__":
    main()