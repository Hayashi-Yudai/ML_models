import sys
sys.path.append('../')

import unittest
import numpy as np
import main

testcase = 1

class TestParser(unittest.TestCase):
  @unittest.skipIf(testcase == 1, "SKIP")
  def test_default_argment(self):
    parser = main.get_parser().parse_args()
    self.assertEqual(parser.epoch, 100)
    self.assertEqual(parser.learning_rate, 0.0001)
    self.assertEqual(parser.train_rate, 0.8)
    self.assertEqual(parser.batch_size, 50)
    self.assertEqual(parser.l2, 0.05)

class TestLoadDataset(unittest.TestCase):
  IMAGE_DIR = '../dataset/raw_images'
  SEG_DIR = '../dataset/segmented_images'

  def setUp(self):
    self.data = main.generate_data(self.IMAGE_DIR, self.SEG_DIR, 1)

  def test_data_type_is_ndarray_in_list(self):
    for img, seg in self.data:
      self.assertEqual(type(img), list)
      self.assertEqual(type(seg), list)
      self.assertEqual(type(img[0]), type(np.array([])))
      self.assertEqual(type(seg[0]), type(np.array([])))

  def test_size_is_valid(self):
    class_num = 2
    for img, seg in self.data:
      self.assertEqual(img[0].shape, (128, 128, 3))
      self.assertEqual(seg[0].shape, (128, 128, class_num))
      self.assertEqual(img[0].dtype, np.float32)
      self.assertEqual(seg[0].dtype, np.int8)

  def test_is_normalized(self):
    for img, seg in self.data:
      self.assertTrue(np.min(img[0]) >= 0.0 and np.max(img[0]) <= 1.0)
      self.assertTrue(np.all(seg[0] >= 0))

  def test_batch_generation(self):
    batch_data = main.generate_data(self.IMAGE_DIR, self.SEG_DIR, 5)
    for img, seg in batch_data:
      self.assertEqual(len(img), 5)
      self.assertEqual(len(seg), 5)

      break

  def test_generate_validation_dataset(self):
    val_data = main.generate_data(self.IMAGE_DIR, '', 1)
    for img, seg in val_data:
      self.assertEqual(type(seg), list)
      self.assertEqual(seg[0], None)

      break

class TestPreprocessing(unittest.TestCase):
  def test_preprocess(self):
    input_img = np.random.randint(0, 255, (1, 128, 128, 3))
    segmented = np.array([
      [0, 0, 1],
      [1, 0, 1],
      [0, 1, 0]
    ])
    processed_img, seg = main.preprocess(input_img, segmented, onehot=True)

    # One-Hot representation of 'segmented'
    seg_predicted = np.array([
      [[1, 0], [1, 0], [0, 1]],
      [[0, 1], [1, 0], [0, 1]],
      [[1, 0], [0, 1], [1, 0]]
    ])

    self.assertTrue(np.min(processed_img) >= 0.0 and np.max(processed_img) <= 1.0)
    self.assertTrue(np.all(seg == seg_predicted))


if __name__ == '__main__':
  unittest.main()
