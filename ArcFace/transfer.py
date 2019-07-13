import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Dropout, Flatten,
                                    Dense, Input, Softmax, Lambda)
from tensorflow.keras.backend import l2_normalize, variable, clip, epsilon
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json, load_model
import numpy as np
from model.archs import ArcFace
from model.prepare_data import generate_images

import argparse
import datetime
import pandas as pd
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-pp", "--param_path", default="./params", type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('-b', '--batch_size', default=10, type=int)
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--lr', '--learning_rate', default=1e-2, type=float)
    parser.add_argument('--decay', default=1e-4, type=float)
    parser.add_argument('--backbone', default="VGG16", type=str)
    args = parser.parse_args()

    return args

def vgg16_arcface(params):
    base_url = f"./params/{params}/"
    model = load_model(
        base_url + "params.hdf5",
        custom_objects={"ArcFace" : ArcFace(m=0.5)}
    )

    return model

def transfer_learning(params, args):
    dir_name = args.param_path + "/{0:%Y%m%d-%H%M%S}".format(datetime.datetime.now())
    os.makedirs(dir_name)
    epochs = args.epochs
    batch = args.batch_size
    opt = args.optimizer
    lr = args.lr
    decay = args.decay
    m = args.penalty
    enhance = args.enhance
    backbone = args.backbone

    num_class = 10
    

    filepath = f"{dir_name}/params.hdf5"
    with open(f"{dir_name}/info.txt", "w") as f:
        f.write(f"number of class: {num_class}\n")
        f.write(f"learning rate: {lr}\n")
        f.write(f"batch size: {batch}\n")
        f.write(f"total epochs: {epochs}\n")
        f.write(f"penalty: {m}\n")
        f.write(f"optimizer: {opt}\n")
        f.write(f"decay rate: {decay}\n")
        f.write(f"enhancement: {enhance}\n")
        f.write(f"backbone network: {backbone}\n")
    
    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="auto"
    )
    model = vgg16_arcface(params)
    model.compile(
        optimizer=Adam(0.01),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.summary()
    
    train_generator = generate_images("../../Images", batch)
    val_generator = generate_images("../../Val-images", 201)
    history = model.fit_generator(
       train_generator,
        steps_per_epoch=30,
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
    transfer_learning("", args)