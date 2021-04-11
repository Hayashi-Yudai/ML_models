import yaml
from ACoL.model import ACoL
import ACoL.prepare_data as prepare_data
import tensorflow as tf


def train(args):
    batch_size = args["batch_size"]
    lr = args["lr"]
    epoch = args["epoch"]

    generator = prepare_data.generate_images(args["train_data"], args["batch_size"])
    val_generator = prepare_data.generate_images(
        args["validation_data"], batch_size, train=False
    )

    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=args["save_params"] + "/params.h5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
    )
    csvLogger = tf.keras.callbacks.CSVLogger(args["save_params"] + "/training.log")

    model = ACoL(args)
    if args["use_param"] is not None:
        model.load_weights(args["use_param"])

    model.compile(
        optimizer=tf.optimizers.Adam(lr),
        loss="categorical_crossentropy",
        # metrics=[tf.keras.metrics.Accuracy()],
        metrics=["accuracy"],
    )
    model.summary()
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

    with open("./ACoL/config.yaml") as f:
        args = yaml.safe_load(f)

    train(args)
