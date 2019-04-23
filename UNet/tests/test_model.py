import os
import sys
sys.path.append('../')

import model

import numpy as np
import tensorflow as tf
import unittest

class TestEachLayers(unittest.TestCase):
  def test_convolution_layer(self):
    inputs = np.ones((3, 128, 128, 3))

    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 128, 128, 3])
    layer = model.conv2d(x, filters=64)
    layer_with_l2 = model.conv2d(x, filters=64, l2_reg=0.01)
    layer_with_bn = model.conv2d(x, filters=64, is_training=True)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
      init.run()
      outputs = sess.run(layer, feed_dict={x: inputs})
      outputs_with_l2 = sess.run(layer_with_l2, feed_dict={x: inputs})
      outputs_with_bn = sess.run(layer_with_bn, feed_dict={x: inputs})

    self.assertEqual(outputs.shape, (3, 128, 128, 64))
    self.assertEqual(outputs_with_l2.shape, (3, 128, 128, 64))
    self.assertEqual(outputs_with_bn.shape, (3, 128, 128, 64))

    self.assertFalse(np.all(outputs == outputs_with_l2))
    self.assertFalse(np.all(outputs == outputs_with_bn))
    self.assertFalse(np.all(outputs_with_l2 == outputs_with_bn))

  def test_transposed_convolution_layer(self):
    inputs = np.ones((3, 128, 128, 3))

    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 128, 128, 3])
    layer = model.trans_conv(x, filters=64)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
      init.run()
      outputs = sess.run(layer, feed_dict={x: inputs})

    self.assertEqual(outputs.shape, (3, 256, 256, 64))

if __name__ == '__main__':
  unittest.main()