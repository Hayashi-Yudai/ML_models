import os
import tensorflow as tf
import tflearn
import numpy as np
from PIL import Image

DIR_NAME = os.path.dirname(os.path.abspath(__file__))

class ACoL:
  def __init__(self, batch_size, class_num, threshold=0.9):
    self.batch_size = batch_size
    self.threshold = threshold
    self.saver = tf.train.import_meta_graph(DIR_NAME + '/vgg16/vgg16.ckpt.meta')

    self.X = tf.get_default_graph().get_tensor_by_name('inputs:0')
    self.y = tf.placeholder(tf.float32, [None, class_num])
    self.conv5_3 = tf.get_default_graph() \
                     .get_tensor_by_name('vgg16/conv5/3/Relu:0')
    self.vgg_stop = tf.stop_gradient(self.conv5_3)
    self.branchA()
    self.branchB()
    self.output_layer()
  

  def branchA(self):
    with tf.name_scope('branch_A'):
      conv1 = tf.layers.conv2d(
        inputs=self.vgg_stop, filters=1024,
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
             self.vgg_stop[batch, :, :, ch],
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
  
  
  def output_layer(self):
    with tf.name_scope('output'):
      self.output = tf.maximum(self.feature_map_A, self.feature_map_B)


  def training(self, inputs, labels):
    # TODO: get hyperparameters from parser
    lossA = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels, logits=self.flattenA
        )
    )
    lossB = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels, logits=self.flattenB
        )
    )
    loss = lossA + lossB

    with tf.name_scope('ops'):
      train_ops = tf.train.AdamOptimizer(0.01).minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
      init.run()
      self.saver.restore(DIR_NAME + '/vgg16/vgg16.ckpt')

      sess.run(train_ops, feed_dict={self.X: inputs, self.y: labels})