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
    self.img = np.random.rand(self.BATCH_SIZE, 224, 224, 3)

    tf.reset_default_graph()
    self.acol = model.ACoL()


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

    self.assertEqual(backborn_shape, (self.BATCH_SIZE, 14, 14, 512))


if __name__ == '__main__':
  unittest.main()