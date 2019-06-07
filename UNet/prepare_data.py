import os
import random
import math

import numpy as np
from PIL import Image, ImageOps, ImageEnhance


def preprocess(img, seg, img_list, seg_list, n_class, onehot):
  """
  Preprocess the image for training or validation

  Parameters
  ==========
  img: image opened by PIL library. Original images
  seg: image opened by PIL library. Segmented images
  img_list, seg_list: list of np.ndarray that contains preprocessed images.
  n_class: int that represents the number of the class
  onehot: bool that determine wheter ones use one-hot representation 

  Returns
  =======
  None
  """
  for i in range(10):
    img2 = img
    seg2 = seg
    mark = [0, 1]
    if seg2 is not None:
      if np.random.choice(mark) == 1:
        img2 = ImageOps.flip(img2)
        seg2 = ImageOps.flip(seg2)
      if np.random.choice(mark) == 1:
        img2 = ImageOps.mirror(img2)
        seg2 = ImageOps.mirror(seg2)

    color_img = ImageEnhance.Color(img2)   
    change = np.random.rand() / 2.0 + 0.75
    img2 = color_img.enhance(change)

    img2 = img2.resize([128, 128])
    img2 = np.asarray(img2, dtype=np.float32)
    if seg2 is not None:
      seg2 = seg2.resize([128, 128])
      seg2 = np.asarray(seg2, dtype=np.int16)

    if onehot and seg is not None:
      identity = np.identity(n_class, dtype=np.int16) 
      seg2 = identity[seg2]
    img_list.append(img2/255.0)
    if seg2 is not None:
      seg_list.append(seg2)

  return None


def load_data(image_dir, seg_dir, n_class, train_val_rate, onehot=True):
  """
  load images and segmented images.

  Parameters
  ==========
  image_dir: string 
    the directory of the raw images.
  seg_dir: string
    the directory of the segmented images.
  n_class: int
    the number of classes
  train_val_rate: float
    the ratio of training data to the validation data
  onehot: bool

  Returns
  =======
  the tuple.(
    (training image, training segmented image),
    (validation image, validation segmented image)
    )
  training/validation (segmented) images are the list of np.ndarray whose shape
  is (128, 128, 3) (row image) and (128, 128, n_class) (segmented image)
  """
  row_img = []
  segmented_img = []
  images = os.listdir(image_dir)
  random.shuffle(images)
  for img in images:
    if img.endswith('.png') or img.endswith('.jpg'):
      split_name = os.path.splitext(img)
      img = Image.open(os.path.join(image_dir, img))
      if seg_dir is not None:
        seg = Image.open(os.path.join(seg_dir, split_name[0] + '-seg' + split_name[1]))
      else:
        seg = None

      preprocess(img, seg, row_img, segmented_img, n_class, onehot=onehot)
  
  train_val_border = int(len(row_img) * train_val_rate)
  train_data = row_img[: train_val_border], \
                segmented_img[: train_val_border]
  validation_data = row_img[train_val_border:],\
                    segmented_img[train_val_border:]

  return train_data, validation_data


def generate_data(input_images, teacher_images, batch_size):
  """
  generate the pair of the raw image and segmented image.

  Parameters
  ========== 
  inputs_images: list of the np.array
  teacher_image: list of the np.array or None
  batch_size: int

  Returns
  =======
  """
  batch_num = math.ceil(len(input_images) / batch_size)
  input_images = np.array_split(input_images, batch_num)
  if np.any(teacher_images is None):
    teacher_images = np.zeros(batch_num)
  else:
    teacher_images = np.array_split(teacher_images, batch_num)

  for i in range(batch_num):
    yield input_images[i], teacher_images[i]

    