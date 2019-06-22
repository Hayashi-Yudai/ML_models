import os
import numpy as np
from PIL import Image

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

    return images


def make_dataset(train=True, val=False) -> tuple:
    images = image_files(train)
    class_idx = 0
    dataset = []
    labels = []
    for category in range(len(images)):
        label = np.array([0]*CLASS_NUM)
        label[class_idx] = 1
        for img in images[category]:
            pil_img = Image.open(IMAGE_DIR + img)
            pil_img = pil_img.resize([224, 224])
            np_img = np.asanyarray(pil_img, dtype=np.float32)
            np_img /= 255.0

            dataset.append(np_img)
            labels.append(label)

        class_idx += 1

    if not val:
      return np.array(dataset)[:-50], np.array(labels)[:-50]
    else:
      return np.array(dataset)[-50:], np.array(labels)[-50:]


def generate_dataset(batch_size: int, train=True, val=False):
    dataset, labels = make_dataset(train)
    perm = np.random.permutation(len(labels))
    dataset, labels = dataset[perm], labels[perm]

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


if __name__ == '__main__':
    print(len(make_dataset(False)[0]))
    data = generate_dataset(50, False)
    for d, l in data:
        print(d.shape)
        print(l.shape)
        break
