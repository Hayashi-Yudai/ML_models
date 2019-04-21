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

  def test_data_type_is_ndarray(self):
    for img, seg in main.generate_data(self.IMAGE_DIR, self.SEG_DIR):
      self.assertEqual(type(img), type(np.array([])))
      self.assertEqual(type(seg), type(np.array([])))

if __name__ == '__main__':
  unittest.main()