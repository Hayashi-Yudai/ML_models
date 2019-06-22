import sys
import os
FILE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE + '/../')

import prepare_data
import numpy as np
import unittest


class TestLoadData(unittest.TestCase):
    def test_datafile(self):
        image_dirs = prepare_data.image_files(train=False)
        self.assertEqual(len(image_dirs), 10)

    def test_load_images(self):
        img, label = prepare_data.make_dataset(train=False)

        self.assertEqual(type(img), np.ndarray)
        self.assertEqual(type(label), np.ndarray)

        self.assertEqual(img[0].shape, (224, 224, 3))
        self.assertEqual(label[0].shape, (10, ))
        self.assertEqual(sum(label[0]), 1)

    def test_generate_dataset(self):
        gen = prepare_data.generate_dataset(batch_size=10, train=False)

        for img, label in gen:
            self.assertEqual(len(img), 10)
            self.assertEqual(len(label), 10)
            break

    def test_validation_dataset(self):
        img, label = prepare_data.make_dataset(train=False, val=True)
        self.assertEqual(img[0].shape, (224, 224, 3))
        self.assertEqual(label[0].shape, (10, ))
        self.assertEqual(sum(label[0]), 1)



if __name__ == '__main__':
    unittest.main()
