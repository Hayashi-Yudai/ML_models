from UNet import train
import numpy as np


def test_dice_coef():
    test_input = np.zeros((1, 3, 3, 3))
    test_input2 = np.ones((1, 3, 3, 3))

    assert train.dice_coef(test_input, test_input) == 0.0
    assert train.dice_coef(test_input2, test_input2) == 54 / 55


def test_dice_loss():
    test_input = np.zeros((1, 3, 3, 3))
    test_input2 = np.ones((1, 3, 3, 3))

    assert train.dice_coef_loss(test_input, test_input) == 1.0
    assert train.dice_coef_loss(test_input2, test_input2) == 1 - 54 / 55
