from ACoL.prepare_data import AugmentImageGenerator, preprocessing, generate_images
import numpy as np


def test_preprocessing_primitives():
    int_input = 255
    float_input = 255.0

    assert preprocessing(int_input) == 1.0
    assert preprocessing(float_input) == 1.0


def test_preprocessing_array():
    test_input = np.ones((2, 2, 3)) * 255

    assert preprocessing(test_input).shape == (2, 2, 3)
    assert np.all(preprocessing(test_input) == np.ones((2, 2, 3)))


def test_generator_val():
    imgs_dir = "/home/yudai/Pictures/raw-img/train"  # has 10 folders
    gen = AugmentImageGenerator()

    width = 30
    height = 30
    batch_size = 5
    flow = gen.flow_from_directory(
        imgs_dir,
        target_size=(height, width),
        batch_size=batch_size,
        class_mode="categorical",
        train=False,
    )

    for bt in flow:
        input_img, label = bt
        assert input_img.shape == (batch_size, height, width, 3)
        assert label.shape == (batch_size, 10)

        assert input_img.dtype == np.float32
        assert label.dtype == np.float32
        assert np.all(input_img <= 255) and np.all(input_img >= 0)

        break


def test_generator_train():
    imgs_dir = "/home/yudai/Pictures/raw-img/train"  # has 10 folders
    gen = AugmentImageGenerator()

    width = 30
    height = 30
    batch_size = 5
    flow = gen.flow_from_directory(
        imgs_dir,
        target_size=(height, width),
        batch_size=batch_size,
        class_mode="categorical",
        train=True,
    )

    for bt in flow:
        input_img, label = bt
        assert input_img.shape == (batch_size, height, width, 3)
        assert label.shape == (batch_size, 10)

        assert input_img.dtype == np.float32
        assert label.dtype == np.float32

        break


def test_generate_images():
    imgs_dir = "/home/yudai/Pictures/raw-img/train"  # has 10 folders
    batch_size = 10

    gen = generate_images(imgs_dir, batch_size)

    for bt in gen:
        input_img, label = bt
        assert input_img.shape == (batch_size, 224, 224, 3)
        assert label.shape == (batch_size, 10)

        assert input_img.dtype == np.float32
        assert label.dtype == np.float32

        break
