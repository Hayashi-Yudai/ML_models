import sys
sys.path.append('../')

import unittest
import numpy as np
import prepare_data 

class TestLoadDataset(unittest.TestCase):
  IMAGE_DIR = '../dataset/raw_images'
  SEG_DIR = '../dataset/segmented_images'

  def setUp(self):
    self.train_data, self.validation_data = \
      prepare_data.load_data(
        self.IMAGE_DIR,
        self.SEG_DIR,
        n_class=2,
        train_val_rate=0.9
        )
    self.data = prepare_data.generate_data(self.IMAGE_DIR, self.SEG_DIR, 1)

  def test_dataset_size(self):
    self.assertEqual(self.train_data[0][0].shape, (128, 128, 3)) # input data
    self.assertEqual(self.train_data[1][0].shape, (128, 128, 2)) # teacher data
    self.assertEqual(self.validation_data[0][0].shape, (128, 128, 3)) # validation data
    self.assertEqual(self.validation_data[1][0].shape, (128, 128, 2)) # validation data

  def test_type_is_valid(self):
    self.assertEqual(type(self.train_data), tuple)
    self.assertEqual(type(self.validation_data), tuple)
    self.assertEqual(type(self.train_data[0]), list)
    self.assertEqual(type(self.validation_data[0]), list)

    self.assertEqual(self.train_data[0][0].dtype, np.float32)
    self.assertEqual(self.train_data[1][0].dtype, np.int16)
    self.assertEqual(self.validation_data[0][0].dtype, np.float32)
    self.assertEqual(self.validation_data[1][0].dtype, np.int16)

  def test_is_normalized(self):
    for img, seg in zip(self.train_data[0], self.train_data[1]):
      self.assertTrue(np.min(img[0]) >= 0.0 and np.max(img[0]) <= 1.0)
      self.assertTrue(np.all(seg[0] >= 0))

  def test_batch_generation(self):
    batch_data = prepare_data.generate_data(*self.train_data, 5)
    for img, seg in batch_data:
      self.assertEqual(len(img), 5)
      self.assertEqual(len(seg), 5)
      break


  def test_generate_test_dataset(self):
    test_data, _ = prepare_data.load_data(self.IMAGE_DIR, None, n_class=2, train_val_rate=0.9)
    test_images = prepare_data.generate_data(*test_data, 1)
    for img, seg in test_images:
      self.assertEqual(type(seg), np.ndarray)
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
    processed_img, seg = prepare_data.preprocess(input_img, segmented, n_class=2, onehot=True)

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