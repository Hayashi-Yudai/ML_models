import os
import tensorflow as tf
import tflearn
import numpy as np
from PIL import Image

DIR_NAME = os.path.dirname(os.path.abspath(__file__))

class ACoL:
  def __init__(self, batch_size, threshold=0.9):
    self.batch_size = batch_size
    self.threshold = threshold
    self.saver = tf.train.import_meta_graph(DIR_NAME + '/vgg16/vgg16.ckpt.meta')

    self.X = tf.get_default_graph().get_tensor_by_name('inputs:0')
    self.conv5_3 = tf.get_default_graph() \
                     .get_tensor_by_name('vgg16/conv5/3/Relu:0')
    self.branchA()
    self.branchB()
  

  def branchA(self):
    with tf.name_scope('branch_A'):
      conv1 = tf.layers.conv2d(
        inputs=self.conv5_3, filters=1024,
        kernel_size=3, strides=1, padding='SAME'
      )
      conv2 = tf.layers.conv2d(
        inputs=conv1, filters=1024,
        kernel_size=3, strides=1, padding='SAME'
      )
      conv3 = tf.layers.conv2d(
        inputs=conv2, filters=10,
        kernel_size=1, strides=1, padding='SAME'
      )
      self.flattenA = tflearn.layers.conv.global_avg_pool(conv3)

      MA = tf.math.argmax(self.flattenA, axis=1)
      max_list = []
      for i in range(self.batch_size):
        max_list.append(conv3[i, :, :, MA[i]])

      self.feature_map_A = tf.stack(max_list)

  def branchB(self):
    with tf.name_scope('branch_B'):
      self.erased = [] 
      for batch in range(self.batch_size):
        tmp = []
        for ch in range(512):
          tmp.append(tf.subtract(
             self.conv5_3[batch, :, :, ch],
             self.feature_map_A[batch]))
        self.erased.append(tf.stack(tmp, axis=2))
      self.erased = tf.stack(self.erased)

      conv1 = tf.layers.conv2d(
        inputs=self.erased, filters=1024,
        kernel_size=3, strides=1, padding='SAME'
      )
      conv2 = tf.layers.conv2d(
        inputs=conv1, filters=1024,
        kernel_size=3, strides=1, padding='SAME'
      )
      conv3 = tf.layers.conv2d(
        inputs=conv2, filters=10,
        kernel_size=1, strides=1, padding='SAME'
      )
      self.flattenB = tflearn.layers.conv.global_avg_pool(conv3)
      MB = tf.math.argmax(self.flattenB, axis=1)
      max_list = []
      for i in range(self.batch_size):
        max_list.append(conv3[i, :, :, MB[i]])

      self.feature_map_B = tf.stack(max_list)