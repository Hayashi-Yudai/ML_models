import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ArcFaceImageGenerator(ImageDataGenerator):
    def __init__(self, **kwargs):
        super(ArcFaceImageGenerator, self).__init__(**kwargs)
    
    def flow_from_directory(self, directory, target_size, batch_size, class_mode):
        batches = super().flow_from_directory(directory, 
            target_size = target_size,
            batch_size = batch_size,
            class_mode = class_mode
        )
        while True:
            inputs, outputs = next(batches)
            yield [inputs, outputs], outputs


def preprocessing(x):
    x -= 127.5
    x /= 128
    
    return x

def generate_images(directory, batch_size):
    train_datagen = ArcFaceImageGenerator(
        preprocessing_function=preprocessing,
        rescale=1.0/255,
        shear_range=0.1,
        zoom_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.04,
        height_shift_range=0.04
    )

    train_generator = train_datagen.flow_from_directory(
        directory,
        target_size=(130, 220),
        batch_size=batch_size,
        class_mode="categorical"
    )

    return train_generator

if __name__ == "__main__":
    pass