import os
import sys
sys.path.append('../')

import model

import numpy as np
import tensorflow as tf
import unittest

class TestLayers(unittest.TestCase):
  def test_convolutional_layer(self):
    inputs = np.ones((3, 128, 128, 3))
    x = tf.placeholder(tf.float32, [None, 128, 128, 3])
    layer = model.conv2d(x, filters=64)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
      init.run()
      outputs = sess.run(layer, feed_dict={x: inputs})
      self.assertEqual(outputs.shape, (3, 128, 128, 64))

if __name__ == '__main__':
  unittest.main()