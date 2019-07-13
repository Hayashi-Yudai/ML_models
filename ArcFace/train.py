import os
import datetime
import numpy as np
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf

from model.archs import resnet50_arcface
from model.prepare_data import generate_images
from model.params_handler import save_info, parse_args


def main(args):
    dir_name = os.path.dirname(os.path.abspath(__file__)) + \
        "/params/{0:%Y%m%d-%H%M%S}".format(datetime.datetime.now())
    os.makedirs(dir_name)
    epochs = args.epochs
    batch = args.batch_size
    lr = args.lr
    decay = args.decay

    filepath = f"{dir_name}/params.hdf5"
    train_generator = generate_images(args.train_data, batch)
    val_generator = generate_images(args.validation_data, 50)
    
    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="auto"
    )
    csvLogger = tf.keras.callbacks.CSVLogger(dir_name + "/training.log")
    model = resnet50_arcface(args)
    model.compile(
        optimizer=SGD(lr, momentum=0.9),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    save_info(dir_name, args, model)
    
    history = model.fit_generator(
       train_generator,
        steps_per_epoch=30,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=8,
        callbacks=[callback, csvLogger]
    )
    

if __name__ == "__main__":
    args = parse_args()
    main(args)