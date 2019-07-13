import os
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import tensorflow as tf

IMAGE_DIR = '/home/yudai/Pictures/raw-img'
CLASS_NUM = 10


def image_files(train=True) -> list:
    suffix = '/train/' if train else '/validation/'
    category_list = os.listdir(IMAGE_DIR + suffix)

    images = []
    for category in category_list:
        tmp = []
        for f in os.listdir(IMAGE_DIR + suffix + category):
            if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg'):
                tmp.append(suffix + f'/{category}/' + f)

        images.append(tmp)
    
    class_idx = 0
    dataset = []
    labels = []
    for category in range(len(images)):
        label = np.array([0]*CLASS_NUM)
        label[class_idx] = 1
        for img in images[category]:
            dataset.append(img)
            labels.append(label)

        class_idx += 1
        
    shuffle = np.random.permutation(len(dataset))
    dataset = np.array(dataset)[shuffle]
    labels = np.array(labels)[shuffle]

    return dataset, labels


def augmentation(img: Image.Image) -> Image.Image:
    mark = [0, 1]
    if np.random.choice(mark) == 1:
        img = ImageOps.flip(img)
    if np.random.choice(mark) == 1:
        img = ImageOps.mirror(img)

    color_img = ImageEnhance.Color(img)
    change = np.random.rand() / 2.0 + 0.75
    img = color_img.enhance(change)

    return img


def make_dataset(train=True, val=False) -> tuple:
    images, labels = image_files(train)

    dataset = []
    answers = []
    if val:
        images = images[-50:]
        labels = labels[-50:]

        for image, label in zip(images, labels):
            pil_img = Image.open(IMAGE_DIR + image)
            pil_img = pil_img.resize([224, 224])
            np_img = np.asanyarray(pil_img, dtype=np.float32)
            np_img /= 255.0
            dataset.append(np_img)
            answers.append(label)
        
        return dataset, answers


    for image, label in zip(images, labels):
        pil_img = Image.open(IMAGE_DIR + image)
        pil_img = pil_img.resize([224, 224])

        for i in range(10):
            pil_img = augmentation(pil_img)

            np_img = np.asanyarray(pil_img, dtype=np.float32)
            np_img /= 255.0
            dataset.append(np_img)
            answers.append(label)
    
    return dataset, answers



def generate_dataset(batch_size: int, train=True, val=False):
    dataset, labels = make_dataset(train)

    bt_dataset = []
    bt_labels = []
    for data, label in zip(dataset, labels):
        if len(bt_dataset) == batch_size:
            yield np.array(bt_dataset), np.array(bt_labels)
            bt_dataset = []
            bt_labels = []
        else:
            bt_dataset.append(data)
            bt_labels.append(label)

####################################################
### Keras
####################################################
def preprocessing(x):
    return (x - 127.5) / 128

def generate_images(directory, batch_size, train=True):
    if train:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocessing,
            shear_range=0.3,
            zoom_range=0.1,
            rotation_range=10,
            fill_mode="constant",
            width_shift_range=0.05,
            height_shift_range=0.05,
        )
    else:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocessing
        )

    generator = datagen.flow_from_directory(
        directory,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="categorical"
    )

    return generator