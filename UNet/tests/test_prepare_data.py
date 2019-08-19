import pytest
import numpy as np
from UNet.prepare_data import data_gen


@pytest.fixture()
def train_data():
    return "./UNet/dataset/raw_images"


@pytest.fixture()
def valid_data():
    return "./UNet/dataset/segmented_images"


def test_data_gen(train_data, valid_data):
    batch_size = 2
    class_num = 2

    generator = data_gen(train_data, valid_data, batch_size, class_num)

    original_img, segment_img = next(generator)

    assert type(original_img) == np.ndarray
    assert type(segment_img) == np.ndarray
    assert original_img.shape == (batch_size, 224, 224, 3)
    assert segment_img.shape == (batch_size, 224, 224, class_num)
    assert np.min(original_img) >= 0.0
    assert np.max(original_img) <= 1.0
