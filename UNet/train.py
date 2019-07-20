import argparse
import tensorflow as tf
import model


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
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    unet.summary()


if __name__ == "__main__":
    args = get_parser().parse_args()
    train(args)
