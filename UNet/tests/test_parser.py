import sys
sys.path.append('../')

import unittest 
import main

class TestParser(unittest.TestCase):
  def testDefaultArgment(self):
    parser = main.get_parser().parse_args()
    self.assertEqual(parser.epoch, 100)
    self.assertEqual(parser.learning_rate, 0.01)
    self.assertEqual(parser.train_rate, 0.8)
    self.assertEqual(parser.batch_size, 50)
    self.assertEqual(parser.l2, 0.001)

if __name__ == '__main__':
  unittest.main()