import os
FILE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(FILE + '/../')

import unittest
import numpy as np
import prepare_data


class TestLoadData(unittest.TestCase):
  def test_datafile(self):
    image_dirs = prepare_data.image_files(train=False)
    self.assertEqual(len(image_dirs), 10)


  def test_load_images(self):
    img = prepare_data.make_dataset(train=False)

    self.assertEqual(type(img), np.ndarray)
    self.assertEqual(type(img[0]), np.ndarray)
    self.assertEqual(img[0][0].shape, (224, 224, 3))
    self.assertEqual(img[0][1].shape, (10,))
    self.assertEqual(sum(img[0][1]), 1)


  def test_generate_dataset(self):
    gen = prepare_data.generate_dataset(batch_size=10, train=False)

    for dataset in gen:
      self.assertEqual(len(dataset), 10)
      break


if __name__ == '__main__':
  unittest.main()