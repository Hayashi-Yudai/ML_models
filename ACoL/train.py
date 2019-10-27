import argparse
from model import ACoL
import prepare_data
import tensorflow as tf
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--train_data", type=str, default="./")
    parser.add_argument("--validation_data", type=str, default="./")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--save_params", type=str, default="./")
    parser.add_argument("--use_param", type=str, default="")

    return parser


def train(args):
    batch_size = args.batch_size
    lr = args.lr
    epoch = args.epoch

    generator = prepare_data.generate_images(args.train_data, batch_size)
    val_generator = prepare_data.generate_images(
        args.validation_data, batch_size, train=False
    )

    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.save_params + "/params.h5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
    )
    csvLogger = tf.keras.callbacks.CSVLogger(args.save_params + "/training.log")

    model = ACoL(args)
    if args.use_param != "":
        model.load_weights(args.use_param)

    model.compile(
        optimizer=tf.train.AdamOptimizer(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(model.summary())
    model.fit_generator(
        generator,
        steps_per_epoch=150,
        epochs=epoch,
        validation_data=val_generator,
        validation_steps=10,
        callbacks=[csvLogger, callback],
    )


if __name__ == "__main__":
    if tf.__version__ >= "2.0.0":
        device = tf.config.experimental.list_physical_devices("GPU")
        if len(device) > 0:
            for dev in device:
                tf.config.experimental.set_memory_growth(dev, True)
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.keras.backend.set_session(tf.Session(config=config))

    parser = get_parser().parse_args()
    train(parser)
