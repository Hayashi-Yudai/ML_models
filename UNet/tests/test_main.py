import os
import sys
sys.path.append('../')

import unittest 
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
  def test_file_exist(self):
    self.assertTrue(os.path.exists('../dataset/raw_images'))
    self.assertTrue(os.path.exists('../dataset/segmented_images'))

if __name__ == '__main__':
  unittest.main()