import argparse
import model


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', type=int, default=100)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
    parser.add_argument('-b', '--batch_size', type=int, default=10)
    parser.add_argument('-th', '--threshold', type=float, default=0.9)

    return parser


if __name__ == '__main__':
    parser = get_parser().parse_args()
    acol = model.ACoL(batch_size=parser.batch_size,
                      class_num=10,
                      threshold=parser.threshold)
    acol.training(parser)
