import argparse
import tensorflow as tf
import tensorflow.keras.backend as K
from UNet import model
from UNet.prepare_data import data_gen


def get_parser():
    """
    Set hyper parameters for training UNet.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument(
        "--train_rate", type=float, default=0.8, help="ratio of training data"
    )
    parser.add_argument("--train_data", type=str, default="./dataset/raw_images/")
    parser.add_argument(
        "--segmented_data", type=str, default="./dataset/segmented_images/"
    )
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--l2", type=float, default=0.05, help="L2 regularization")
    parser.add_argument("--weights", default="", type=str)

    return parser


def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return 2.0 * intersection / (K.sum(y_true) + K.sum(y_pred) + 1)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def train(args: "argparse.Namespace"):
    lr: float = args.learning_rate
    n_classes: int = args.n_classes

    unet = model.UNet(args)
    unet.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=dice_coef_loss,
        metrics=["accuracy"],
    )
    unet.summary()

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath="./params/model.h5",
        monitor="loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    segmented_data = (
        args.segmented_data
    )  # os.path.join(args.train_data, "../segmented_images")
    generator = data_gen(args.train_data, segmented_data, args.batch_size, n_classes)
    unet.fit_generator(generator, steps_per_epoch=30, epochs=100, callbacks=[ckpt])


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

    args = get_parser().parse_args()
    train(args)
