import os
import tensorflow as tf
import tflearn
import numpy as np
from PIL import Image

DIR_NAME = os.path.dirname(os.path.abspath(__file__))

class ACoL:
  def __init__(self):
    self.saver = tf.train.import_meta_graph(DIR_NAME + '/vgg16/vgg16.ckpt.meta')

    self.X = tf.get_default_graph().get_tensor_by_name('inputs:0')
    self.conv5_3 = tf.get_default_graph() \
                      .get_tensor_by_name('vgg16/conv5/3/Relu:0')

"""
with tf.name_scope('branch_A'):
  conv1 = tf.layers.conv2d(
    inputs=conv5_3,
    filters=1024,
    kernel_size=3,
    strides=1,
    padding='SAME'
  )
  conv2 = tf.layers.conv2d(
    inputs=conv1,
    filters=1024,
    kernel_size=3,
    strides=1,
    padding='SAME'
  )
  conv3 = tf.layers.conv2d(
    inputs=conv2,
    filters=10,
    kernel_size=1,
    strides=1,
    padding='SAME'
  )
  flatten = tflearn.layers.conv.global_avg_pool(conv3)
  M = tf.math.argmax(flatten, axis=1)
  max_list = []
  for i in range(2):
    max_list.append(conv3[i, M[i]])

  feature_map_A = tf.stack(max_list)

with tf.name_scope('branch_B'):
  threshold = 0.9
"""