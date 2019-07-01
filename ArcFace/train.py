import os
import datetime
import pandas as pd
from PIL import Image
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import argparse

from model.archs import vgg16_arcface
from model.prepare_data import generate_images

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-pp", "--param_path", default="./lab-cardimage-match/params", type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('-b', '--batch_size', default=10, type=int)
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--lr', '--learning_rate', default=1e-2, type=float)
    parser.add_argument('--decay', default=1e-4, type=float)
    parser.add_argument('--backbone', default="VGG16", type=str)
    parser.add_argument('--fine', default="", type=str)
    args = parser.parse_args()

    return args

def main(args):
    dir_name = args.param_path + "/{0:%Y%m%d-%H%M%S}".format(datetime.datetime.now())
    params_base = "./lab-cardimage-match/params/"
    os.makedirs(dir_name)
    epochs = args.epochs
    batch = args.batch_size
    opt = args.optimizer
    lr = args.lr
    decay = args.decay
    backbone = args.backbone
    fine_tune = None if args.fine == "" else params_base + args.fine + "/params.hdf5"

    num_class = 10
    m = 0.0

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
        mode="auto"
    )
    model = vgg16_arcface(num_class, m, decay, fine_tune)
    model.compile(
        optimizer=Adam(0.01),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    
    train_generator = generate_images("./sample-images", batch)
    val_generator = generate_images("./validation-images", 201)
    history = model.fit_generator(
       train_generator,
        steps_per_epoch=20,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=1,
        callbacks=[callback]
    )
    df = pd.DataFrame({"val_loss" : history.history["val_loss"], "val_acc" : history.history["val_acc"], 
                       "loss" : history.history["loss"], "acc" : history.history["acc"]})
    df.to_csv(dir_name + "/history.csv", index=False)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)

