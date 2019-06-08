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


def make_dataset(train=True) -> np.ndarray:
  images = image_files(train)
  class_idx = 0
  dataset = []
  for category in range(len(images)):
    label = np.array([0]*CLASS_NUM)
    label[class_idx] = 1
    for img in images[category]:
      pil_img = Image.open(IMAGE_DIR + img)
      pil_img = pil_img.resize([224, 224])
      np_img = np.asanyarray(pil_img, dtype=np.float32)
      np_img /= 255.0

      dataset.append((np_img, label))

    class_idx += 1

  return np.array(dataset)


def generate_dataset(batch_size: int, train=True):
  dataset = make_dataset(train)
  np.random.shuffle(dataset)
  print(len(dataset))
  bt_dataset = []
  for data in dataset:
    if len(bt_dataset) == batch_size:
      yield bt_dataset
      bt_dataset = []
    else:
      bt_dataset.append(data)

if __name__ == '__main__':
  print(type(make_dataset(False)[0]))