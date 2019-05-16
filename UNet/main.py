import argparse
import os
import glob
import random
import math

import numpy as np
from PIL import Image

import model

def get_parser():
  """
  Set hyper parameters for training UNet.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('-e', '--epoch', type=int, default=100)
  parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
  parser.add_argument('-tr', '--train_rate', type=float, default=0.8,
                      help='ratio of training data')
  parser.add_argument('-b', '--batch_size', type=int, default=20)
  parser.add_argument('-l2', '--l2', type=float, default=0.05,
                      help='L2 regularization')

  return parser


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
  for idx, img in enumerate(images):
    if img.endswith('.png') or img.endswith('.jpg'):
      split_name = os.path.splitext(img)
      img = Image.open(os.path.join(image_dir, img))
      if seg_dir != '':
        seg = Image.open(os.path.join(seg_dir, split_name[0] + '-seg' + split_name[1]))
        seg = seg.resize((128, 128)) 
        seg = np.asarray(seg, dtype=np.int16)
      else:
        seg = None

      img = img.resize((128, 128))
      img = np.asarray(img, dtype=np.float32)
      img, seg = preprocess(img, seg, n_class, onehot=onehot)
      row_img.append(img)
      segmented_img.append(seg)
  
  train_data = row_img[: int(len(row_img) * train_val_rate)], \
                segmented_img[:int(len(row_img)*train_val_rate)]
  validation_data = row_img[int(len(row_img) * train_val_rate) :],\
                    segmented_img[int(len(row_img) * train_val_rate):]

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
  if np.any(teacher_images == None):
    teacher_images = np.zeros(batch_num)
  else:
    teacher_images = np.array_split(teacher_images, batch_num)

  for i in range(batch_num):
    yield input_images[i], teacher_images[i]

    
def preprocess(img, seg, n_class, onehot):
  if onehot and seg is not None:
    identity = np.identity(n_class, dtype=np.int16) 
    seg = identity[seg]

  return img / 255.0, seg


if __name__ == '__main__':
  parser = get_parser().parse_args()
  unet = model.UNet(classes=2)
  unet.train(parser)