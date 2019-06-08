import os
FILE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(FILE + '/../')

import unittest
import prepare_data


class TestLoadData(unittest.TestCase):
  def test_datafile(self):
    image_dirs = prepare_data.image_files()
    self.assertEqual(len(image_dirs), 10)


if __name__ == '__main__':
  unittest.main()