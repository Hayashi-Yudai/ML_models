import argparse
import model
import prepare_data
import tensorflow as tf


def get_parser():
    """
    Set hyper parameters for training UNet.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument(
        "--train_rate", type=float, default=0.8, help="ratio of training data"
    )
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--l2", type=float, default=0.05, help="L2 regularization")

    return parser.parse_args()


def loss(y_true, y_pred):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)

    return tf.reduce_sum(-tf.log(tf.clip_by_value(y_pred, 1e-10, 1.0)) * y_true)


def train(args):
    lr = args.learning_rate
    unet = model.UNet(args)
    unet.compile(
        optimizer=tf.keras.optimizers.Adam(lr), loss=loss, metrics=["accuracy"]
    )
    unet.summary()

    generator = prepare_data.data_gen(
        "./dataset/raw_images/", "./dataset/segmented_images/", 2
    )
    unet.fit_generator(generator, steps_per_epoch=100, epochs=10)


if __name__ == "__main__":
    args = get_parser()
    train(args)
