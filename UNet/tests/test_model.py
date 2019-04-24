import os
import sys
sys.path.append('../')

import model

import numpy as np
import tensorflow as tf
import unittest

class TestEachLayers(unittest.TestCase):
  def setUp(self):
    self.inputs = np.ones((3, 128, 128, 3))

  def test_convolution_layer(self):
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 128, 128, 3])
    layer = model.conv2d(x, filters=64)
    layer_with_l2 = model.conv2d(x, filters=64, l2_reg=0.01)
    layer_with_bn = model.conv2d(x, filters=64, is_training=True)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
      init.run()
      outputs = sess.run(layer, feed_dict={x: self.inputs})
      outputs_with_l2 = sess.run(layer_with_l2, feed_dict={x: self.inputs})
      outputs_with_bn = sess.run(layer_with_bn, feed_dict={x: self.inputs})

    self.assertEqual(outputs.shape, (3, 128, 128, 64))
    self.assertEqual(outputs_with_l2.shape, (3, 128, 128, 64))
    self.assertEqual(outputs_with_bn.shape, (3, 128, 128, 64))

    self.assertFalse(np.all(outputs == outputs_with_l2))
    self.assertFalse(np.all(outputs == outputs_with_bn))
    self.assertFalse(np.all(outputs_with_l2 == outputs_with_bn))

  def test_transposed_convolution_layer(self):
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 128, 128, 3])
    layer = model.trans_conv(x, filters=64)
    layer_with_l2 = model.trans_conv(x, filters=64, l2_reg=0.01)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
      init.run()
      outputs = sess.run(layer, feed_dict={x: self.inputs})
      outputs_with_l2 = sess.run(layer_with_l2, feed_dict={x: self.inputs})

    self.assertEqual(outputs.shape, (3, 256, 256, 64))
    self.assertEqual(outputs_with_l2.shape, (3, 256, 256, 64))
    self.assertFalse(np.all(outputs == outputs_with_l2))

  def test_pooling_layer(self):
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 128, 128, 3])
    layer = model.pooling(x)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
      init.run()
      outputs = sess.run(layer, feed_dict={x: self.inputs})

    self.assertEqual(outputs.shape, (3, 64, 64, 3))

class TestUNet(unittest.TestCase):
  def test_unet(self):
    inputs = np.ones((3, 128, 128, 3))

    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 128, 128, 3])
    layer = model.UNet(x, classes=2, l2_reg=0.01, is_training=True)
    layer_val = model.UNet(x, classes=2, l2_reg=0.01, is_training=False)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
      init.run()
      outputs = sess.run(layer, feed_dict={x: inputs})
      outputs_val = sess.run(layer, feed_dict={x: inputs})

    self.assertEqual(outputs.shape, (3, 128, 128, 2))
    self.assertEqual(outputs_val.shape, (3, 128, 128, 2))

if __name__ == '__main__':
  unittest.main()