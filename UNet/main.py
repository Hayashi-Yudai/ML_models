import argparse
import os
import glob

import numpy as np
from PIL import Image

import model

def get_parser():
  """
  Set hyper parameters for training UNet.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('-e', '--epoch', type=int, default=100)
  parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
  parser.add_argument('-tr', '--train_rate', type=float, default=0.8, help='ratio of training data')
  parser.add_argument('-b', '--batch_size', type=int, default=50)
  parser.add_argument('-l2', '--l2', type=float, default=0.001, help='L2 regularization')

  return parser

def generate_data(image_dir, seg_dir, onehot=True):
  """
  generate the pair of the raw image and segmented image.
  Args:
    image_dir: the directory of the raw images.
    seg_dir: the directory of the segmented images.
  Returns:
    yield two np.ndarrays. The shapes are (128, 128, 3) and (128, 128)
  """
  #TODO: create batch by cropping and augumentation.
  for img in os.listdir(image_dir):
    if img.endswith('.png') or img.endswith('.jpg'):
      split_name = os.path.splitext(img)
      img = Image.open(os.path.join(image_dir, img))
      seg = Image.open(os.path.join(seg_dir, split_name[0] + '-seg' + split_name[1]))

      img = img.resize((128, 128))
      seg = seg.resize((128, 128)) 

      img = np.asarray(img, dtype=np.float32)
      seg = np.asarray(seg, dtype=np.int8)

      img, seg = preprocess(img, seg, onehot=onehot)
      yield img, seg
    
def preprocess(img, seg, onehot):
  if onehot:
    identity = np.identity(2, dtype=np.int8)  #TODO: the number of class is hard coded
    seg = identity[seg]

  return img / 255.0, seg

if __name__ == '__main__':
  parser = get_parser.parse_args()