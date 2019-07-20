import sys

sys.path.append("../")

import unittest
import numpy as np
import main

testcase = 0


class TestParser(unittest.TestCase):
    @unittest.skipIf(testcase == 1, "SKIP")
    def test_default_argment(self):
        parser = main.get_parser().parse_args()
        self.assertEqual(parser.epoch, 100)
        self.assertEqual(parser.learning_rate, 0.0001)
        self.assertEqual(parser.train_rate, 0.8)
        self.assertEqual(parser.batch_size, 20)
        self.assertEqual(parser.l2, 0.05)


if __name__ == "__main__":
    unittest.main()
