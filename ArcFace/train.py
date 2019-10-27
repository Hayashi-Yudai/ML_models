import os
import datetime
import numpy as np
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf

from model.archs import arcface_main
from model.prepare_data import generate_images
from model.params_handler import save_info, parse_args


def main(args):
    dir_name = args.save_path + "/params/{0:%Y%m%d-%H%M%S}".format(
        datetime.datetime.now()
    )
    os.makedirs(dir_name)
    epochs = args.epochs
    batch = args.batch_size
    lr = args.lr
    if args.optimizer == "Adam":
        optimizer = Adam(lr)
    else:
        optimizer = SGD(lr, momentum=0.9, nesterov=True)

    filepath = f"{dir_name}/params.hdf5"
    train_generator = generate_images(args.train_data, batch)
    val_generator = generate_images(args.validation_data, 50, train=False)

    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
    )
    csvLogger = tf.keras.callbacks.CSVLogger(dir_name + "/training.log")
    scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.9, patience=5, min_delta=0.002
    )
    model = arcface_main(args)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.summary()
    save_info(dir_name, args, model)

    model.fit_generator(
        train_generator,
        steps_per_epoch=400,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=10,
        callbacks=[callback, csvLogger, scheduler],
    )


if __name__ == "__main__":
    device = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(device[0], True)

    args = parse_args()
    main(args)
