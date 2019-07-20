import argparse
import model
import prepare_data
import tensorflow as tf


class DiceLossByClass:
    def __init__(self, input_shape, class_num):
        self.__input_h = input_shape[0]
        self.__input_w = input_shape[1]
        self.__class_num = class_num

    def dice_coef(self, y_true, y_pred):
        y_true = tf.keras.backend.flatten(y_true)
        y_pred = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true * y_pred)
        denominator = tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred)
        if denominator == 0:
            return 1
        if intersection == 0:
            return 1 / (denominator + 1)
        return (2.0 * intersection) / denominator

    def dice_coef_loss(self, y_true, y_pred):
        # (N, h, w, ch)
        y_true_res = tf.reshape(
            y_true, (-1, self.__input_h, self.__input_w, self.__class_num)
        )
        y_pred_res = tf.reshape(
            y_pred, (-1, self.__input_h, self.__input_w, self.__class_num)
        )
        y_trues = tf.unstack(y_true_res, axis=3)
        y_preds = tf.unstack(y_pred_res, axis=3)

        losses = []
        for y_t, y_p in zip(y_trues, y_preds):
            losses.append((1 - self.dice_coef(y_t, y_p)) * 3)

        return tf.reduce_sum(tf.stack(losses))


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


def train(args):
    lr = args.learning_rate
    unet = model.UNet(args)
    unet.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=DiceLossByClass((388, 388), 2).dice_coef_loss,
        metrics=["accuracy"],
    )
    unet.summary()

    generator = prepare_data.data_gen(
        "./dataset/raw_images/", "./dataset/segmented_images/", 2
    )
    unet.fit_generator(generator, steps_per_epoch=100, epochs=10)


if __name__ == "__main__":
    args = get_parser()
    train(args)
