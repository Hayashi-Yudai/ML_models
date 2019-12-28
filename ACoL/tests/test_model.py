import ACoL.model as model
import pytest
import numpy as np
import tensorflow as tf


@pytest.fixture(scope="module")
def setup():
    devices = tf.config.experimental.list_physical_devices("GPU")
    if devices:
        for dev in devices:
            tf.config.experimental.set_memory_growth(dev, True)


@pytest.fixture
def params():
    class p:
        n_classes = 10
        batch_size = 1
        threshold = 0.5

    return p()


def test_subbranch(setup, params):
    width = 10
    height = 10
    inputs = np.random.random((1, height, width, 3))
    branch = model.subbranch()

    output = branch(inputs)
    assert len(output) == 2

    x, features = output
    assert x.shape == (1, params.n_classes)
    assert features.shape == (1, height, width, params.n_classes)
