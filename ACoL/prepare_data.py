import tensorflow as tf
import imgaug.augmenters as iaa


class AugmentImageGenerator(tf.keras.preprocessing.image.ImageDataGenerator):
    def __init__(self, **kwargs):
        super(AugmentImageGenerator, self).__init__(**kwargs)

    def flow_from_directory(
        self, directory, target_size, batch_size, class_mode, train=True
    ):
        batches = super().flow_from_directory(
            directory,
            target_size=target_size,
            batch_size=batch_size,
            class_mode=class_mode,
        )
        seq = iaa.Sequential(
            [
                iaa.GaussianBlur(sigma=(0, 1)),
                iaa.ChannelShuffle(p=0.5),
                iaa.Add(value=(-0.3, 0.3)),
            ]
        )

        while True:
            inputs, outputs = next(batches)
            if train:
                inputs = seq.augment_images(inputs)

            yield inputs, outputs


def preprocessing(x):
    return (x - 127.5) / 128


def generate_images(directory, batch_size, train=True):
    if train:
        datagen = AugmentImageGenerator(
            preprocessing_function=preprocessing,
            shear_range=0.3,
            zoom_range=0.2,
            rotation_range=15,
            fill_mode="constant",
            width_shift_range=0.1,
            height_shift_range=0.1,
        )
    else:
        datagen = AugmentImageGenerator(preprocessing_function=preprocessing)

    generator = datagen.flow_from_directory(
        directory,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="categorical",
        train=False,
    )

    return generator
