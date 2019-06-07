import os
import sys
sys.path.append('../')

import unittest
import numpy as np
from PIL import Image
import tensorflow as tf

import model

class TestModel(unittest.TestCase):
  def setUp(self):
    self.CKPT_PATH = os.path.dirname(os.path.abspath(__file__)) + '/../vgg16/'
    self.BATCH_SIZE = 5
    self.CLASS_NUM = 10
    self.img = np.random.rand(self.BATCH_SIZE, 224, 224, 3)

    tf.reset_default_graph()
    self.acol = model.ACoL(self.BATCH_SIZE)


  def test_shapes(self):
    saver = tf.train.import_meta_graph(self.CKPT_PATH + 'vgg16.ckpt.meta')
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
      init.run()
      saver.restore(sess, self.CKPT_PATH + 'vgg16.ckpt')
      backborn_shape = sess.run(
        self.acol.conv5_3,
        feed_dict={self.acol.X: self.img}
      ).shape

      # Branch A
      flattenA_shape = sess.run(
        self.acol.flattenA,
        feed_dict={self.acol.X: self.img}
      ).shape
      feature_map_A_shape = sess.run(
        self.acol.feature_map_A,
        feed_dict={self.acol.X: self.img}
      ).shape

      # Branch B
      flattenB_shape = sess.run(
        self.acol.flattenB,
        feed_dict={self.acol.X: self.img}
      ).shape
      feature_map_B_shape = sess.run(
        self.acol.feature_map_B,
        feed_dict={self.acol.X: self.img}
      ).shape

    self.assertEqual(backborn_shape, (self.BATCH_SIZE, 14, 14, 512))
    self.assertEqual(flattenA_shape, (self.BATCH_SIZE, self.CLASS_NUM))
    self.assertEqual(feature_map_A_shape, (self.BATCH_SIZE, 14, 14))
    self.assertEqual(flattenB_shape, (self.BATCH_SIZE, self.CLASS_NUM))
    self.assertEqual(feature_map_B_shape, (self.BATCH_SIZE, 14, 14))


if __name__ == '__main__':
  unittest.main()