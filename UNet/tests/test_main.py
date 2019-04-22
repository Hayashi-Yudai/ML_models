import os
import shutil
import sys
sys.path.append('../')

import unittest
import numpy as np
import main

class TestParser(unittest.TestCase):
  def test_default_argment(self):
    parser = main.get_parser().parse_args()
    self.assertEqual(parser.epoch, 100)
    self.assertEqual(parser.learning_rate, 0.01)
    self.assertEqual(parser.train_rate, 0.8)
    self.assertEqual(parser.batch_size, 50)
    self.assertEqual(parser.l2, 0.001)

class TestLoadDataset(unittest.TestCase):
  IMAGE_DIR = '../dataset/raw_images'
  SEG_DIR = '../dataset/segmented_images'

  def setUp(self):
    self.data = main.generate_data(self.IMAGE_DIR, self.SEG_DIR)

  def test_data_type_is_ndarray(self):
    for img, seg in self.data:
      self.assertEqual(type(img), type(np.array([])))
      self.assertEqual(type(seg), type(np.array([])))

  def test_size_is_valid(self):
    for img, seg in self.data:
      self.assertEqual(img.shape, (128, 128, 3))
      self.assertEqual(seg.shape, (128, 128))
      self.assertEqual(type(img[0,0,0]), np.float32)
      self.assertEqual(type(seg[0,0]), np.int8)

  def test_is_normalized(self):
    for img, seg in self.data:
      self.assertTrue(np.min(img) >= 0.0 and np.max(img) <= 1.0)
      self.assertTrue(np.all(seg >= 0))

if __name__ == '__main__':
  unittest.main()
