import UNet.model as model
import numpy as np
import tensorflow as tf
import pytest


@pytest.fixture(scope="module")
def setup():
    devices = tf.config.experimental.list_physical_devices("GPU")
    if devices:
        for dev in devices:
            tf.config.experimental.set_memory_growth(dev, True)


@pytest.fixture
def params():
    return {"n_classes": 2, "l2": 0.01, "weights": ""}


def test_layers_callable():
    conv_set = model.conv_set(filters=10)
    upsampling_1 = model.upsampling(filters=10)
    upsampling_2 = model.upsampling(filters=10, cut=5)

    assert callable(conv_set)
    assert callable(upsampling_1)
    assert callable(upsampling_2)


def test_conv_set_output(setup):
    layer = model.conv_set(filters=10)
    test_input = np.random.random((1, 5, 5, 3))

    output = layer(test_input)

    assert output.shape == (1, 5, 5, 10)


def test_upsampling_output(setup):
    layer1 = model.upsampling(filters=10)
    layer2 = model.upsampling(filters=10, cut=1)

    test_input_1_1 = np.random.random((1, 10, 10, 3))
    test_input_1_2 = np.random.random((1, 20, 20, 10))
    test_input_2_2 = np.random.random((1, 22, 22, 10))

    output1 = layer1([test_input_1_1, test_input_1_2])
    output2 = layer2([test_input_1_1, test_input_2_2])

    assert output1.shape == (1, 20, 20, 20)
    assert output2.shape == (1, 20, 20, 20)


def test_unet(setup, params):
    test_input = np.random.random((1, 224, 224, 3))

    unet = model.UNet(params)
    output = unet.predict(test_input)

    assert output.shape == (1, 224, 224, params["n_classes"])


def test_unet_regularizer(setup, params):
    unet = model.UNet(params)

    for layer in unet.layers:
        if "kernel_regularizer" in layer.__dict__:
            assert abs(float(layer.kernel_regularizer.l2) - params["l2"]) < 1e-6
