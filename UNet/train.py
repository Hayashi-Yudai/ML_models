import argparse
import tensorflow as tf
import model
from prepare_data import data_gen


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
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--l2", type=float, default=0.05, help="L2 regularization")

    return parser


def train(args):
    lr = args.learning_rate

    unet = model.UNet(args)
    unet.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    unet.summary()

    generator = data_gen("./dataset/raw_images/", "./dataset/segmented_images/", 4)
    unet.fit_generator(generator, steps_per_epoch=30, epochs=100)


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))

    args = get_parser().parse_args()
    train(args)
