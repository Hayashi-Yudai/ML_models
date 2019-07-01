import os
import datetime
import pandas as pd
from PIL import Image
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import argparse

from archs import vgg16_arcface
from prepare_data import generate_images


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-pp", "--param_path",
                        default="./params", type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('-b', '--batch_size', default=10, type=int)
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--lr', '--learning_rate', default=1e-2, type=float)
    parser.add_argument('--decay', default=1e-4, type=float)
    parser.add_argument('--backbone', default="VGG16", type=str)
    args = parser.parse_args()

    return args


def main(args):
    dir_name = args.param_path + \
        "/{0:%Y%m%d-%H%M%S}".format(datetime.datetime.now())
    os.makedirs(dir_name)
    epochs = args.epochs
    batch = args.batch_size
    opt = args.optimizer
    lr = args.lr
    decay = args.decay
    backbone = args.backbone

    num_class = 10
    m = 0.5

    filepath = f"{dir_name}/params.hdf5"
    with open(f"{dir_name}/info.txt", "w") as f:
        f.write(f"number of class: {num_class}\n")
        f.write(f"learning rate: {lr}\n")
        f.write(f"batch size: {batch}\n")
        f.write(f"total epochs: {epochs}\n")
        f.write(f"penalty: {m}\n")
        f.write(f"optimizer: {opt}\n")
        f.write(f"decay rate: {decay}\n")
        f.write(f"backbone network: {backbone}\n")

    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="auto"
    )
    n_classes = 10
    penalty = 0.5
    trainfile = "/Users/wantedly150/Documents/lab-cardimage-match/lab-cardimage-match/sample-images"
    validationfile = "/Users/wantedly150/Documents/lab-cardimage-match/lab-cardimage-match/validation-images"
    train = generate_images(trainfile, 10)
    val = generate_images(validationfile, 10)

    model = vgg16_arcface(n_classes, penalty, 1e-4)
    model.summary()

    history = model.fit_generator(
        train,
        steps_per_epoch=20,
        epochs=epochs,
        validation_data=val,
        validation_steps=1,
        callbacks=[callback]
    )
    df = pd.DataFrame({"val_loss": history.history["val_loss"], "val_acc": history.history["val_acc"],
                       "loss": history.history["loss"], "acc": history.history["acc"]})
    df.to_csv(dir_name + "/history.csv", index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
