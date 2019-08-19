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
    batch_size = 10
    generator = data_gen(train_data, valid_data, batch_size)

    original_img, segment_img = next(generator)

    assert type(original_img) == np.ndarray
    assert type(segment_img) == np.ndarray
