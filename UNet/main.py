import argparse
import model


def get_parser():
    """
  Set hyper parameters for training UNet.
  """
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch", type=int, default=100)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001)
    parser.add_argument(
        "-tr", "--train_rate", type=float, default=0.8, help="ratio of training data"
    )
    parser.add_argument("-b", "--batch_size", type=int, default=20)
    parser.add_argument(
        "-l2", "--l2", type=float, default=0.05, help="L2 regularization"
    )

    return parser


if __name__ == "__main__":
    parser = get_parser().parse_args()
    unet = model.UNet(classes=2)
    unet.train(parser)
