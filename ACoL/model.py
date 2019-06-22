import os
import tensorflow as tf
import tflearn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import interpolate

import prepare_data

DIR_NAME = os.path.dirname(os.path.abspath(__file__))


class ACoL:
    def __init__(self, batch_size, class_num, threshold=0.9):
        tf.reset_default_graph()
        self.batch_size = batch_size
        self.threshold = threshold
        self.is_training = tf.placeholder(tf.bool)
        self.saver = tf.train.import_meta_graph(
            DIR_NAME + '/vgg16/vgg16.ckpt.meta')

        self.X = tf.get_default_graph().get_tensor_by_name('inputs:0')
        self.y = tf.placeholder(tf.int16, [None, class_num])

    def network(self, is_training):
        self.conv5_3 = tf.get_default_graph() \
                         .get_tensor_by_name('vgg16/conv5/3/Relu:0')
        self.vgg_stop = tf.stop_gradient(self.conv5_3)
        self.branchA(is_training=is_training)
        self.branchB(is_training=is_training)
        self.output_layer()

    @staticmethod
    def conv_with_bn(inputs, filters, ks, st, is_training):
        layer = tf.layers.conv2d(
          inputs=inputs, filters=filters,
          kernel_size=ks, strides=st, padding='SAME'
        )
        if is_training is not None:
          layer = tf.layers.batch_normalization(
            inputs=layer,
            axis=-1,
            momentum=0.9,
            epsilon=0.001,
            center=True,
            scale=True,
            training=is_training
          )

        return layer

    def branchA(self, is_training=None):
        with tf.name_scope('branch_A'):
            conv1A = self.conv_with_bn(self.vgg_stop, 1024, 3, 1, is_training)
            conv2A = self.conv_with_bn(conv1A, 1024, 3, 1, is_training)
            conv3A = self.conv_with_bn(conv2A, 10, 1, 1, is_training)
            self.flattenA = tflearn.layers.conv.global_avg_pool(conv3A)

            MA = tf.math.argmax(self.flattenA, axis=1)
            max_list = []
            for i in range(self.batch_size):
                max_list.append(conv3A[i, :, :, MA[i]])

            self.feature_map_A = tf.stack(max_list)

    def branchB(self, is_training=None):
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

            
            conv1B = self.conv_with_bn(self.erased, 1024, 3, 1, is_training)
            conv2B = self.conv_with_bn(conv1B, 1024, 3, 1, is_training)
            conv3B = self.conv_with_bn(conv2B, 10, 1, 1, is_training)
            self.flattenB = tflearn.layers.conv.global_avg_pool(conv3B)
            MB = tf.math.argmax(self.flattenB, axis=1)
            max_list = []
            for i in range(self.batch_size):
                max_list.append(conv3B[i, :, :, MB[i]])

            self.feature_map_B = tf.stack(max_list)

    def output_layer(self):
        with tf.name_scope('output'):
            self.output = tf.maximum(self.feature_map_A, self.feature_map_B)

    def training(self, parser, is_training):
        lr = parser.learning_rate
        epoch = parser.epoch
        self.network(is_training=True)

        lossA = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.y, logits=self.flattenA
            )
        )
        lossB = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.y, logits=self.flattenB
            )
        )
        loss = lossA + lossB

        train_ops = tf.train.AdamOptimizer(lr).minimize(loss)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            init.run()
            self.saver.restore(sess, DIR_NAME + '/vgg16/vgg16.ckpt')
            for e in range(epoch):
                dataset = prepare_data.generate_dataset(self.batch_size, False)
                for data, label in dataset:
                    sess.run(
                        train_ops,
                        feed_dict={
                          self.X: data,
                          self.y: label,
                          self.is_training: True
                        }
                    )
                    ls = loss.eval(feed_dict={self.X: data, self.y: label})
                print(f"loss#{e}: {ls}")
            
            self.validation(sess)

    def validation(self, sess):
      dataset = prepare_data.generate_dataset(self.batch_size, False)
      for data, _ in dataset:
        origin = data
        res = sess.run(
          self.output,
          feed_dict={
            self.X: data,
            self.is_training: None 
          }
        )
        break

      res = res[0]
      space = int((16*14 - 2) / (14 - 1))
      x = [space*i for i in range(14)]
      y = [space*i for i in range(14)]
      x[-1] = 223
      y[-1] = 223
      f = interpolate.interp2d(x, y, res)
      xx = [i for i in range(224)]
      yy = [i for i in range(224)]
      res = f(xx, yy)

      plt.figure()
      plt.imshow((origin[0]*255).astype(np.int16))
      plt.imshow(res, cmap='seismic', alpha=0.3)
      plt.colorbar()
      plt.show()
