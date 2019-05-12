import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import main

class UNet:
  def __init__(self, classes):
    self.IMAGE_DIR = './dataset/raw_images'
    self.SEGMENTED_DIR = './dataset/segmented_images'
    self.VALIDATION_DIR = './dataset/validation'
    self.classes = classes
    self.X = tf.placeholder(tf.float32, [None, 128, 128, 3]) 
    self.y = tf.placeholder(tf.int16, [None, 128, 128, self.classes])
    self.is_training = tf.placeholder(tf.bool)

  @staticmethod
  def conv2d(
    inputs, filters, kernel_size=3, activation=tf.nn.relu, l2_reg=None, 
    momentum=0.9, epsilon=0.001, is_training=False,
    ):
    """
    convolutional layer. If the l2_reg is a float number, L2 regularization is imposed.
    
    Parameters
    ----------
      inputs: tf.Tensor
      filters: Non-zero positive integer
        The number of the filter 
      activation: 
        The activation function. The default is tf.nn.relu
      l2_reg: None or float
        The strengthen of the L2 regularization
      is_training: tf.bool
        The default is False. If True, the batch normalization layer is added.
      momentum: float
        The hyper parameter of the batch normalization layer
      epsilon: float
        The hyper parameter of the batch normalization layer

    Returns
    -------
      layer: tf.Tensor
    """
    regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg) if l2_reg is not None else None
    layer = tf.layers.conv2d(
      inputs=inputs,
      filters=filters,
      kernel_size=kernel_size,
      padding='SAME',
      activation=activation,
      kernel_regularizer=regularizer
    )

    if is_training is not None:
      layer = tf.layers.batch_normalization(
        inputs=layer,
        axis=-1,
        momentum=momentum,
        epsilon=epsilon,
        center=True,
        scale=True,
        training=is_training
      )

    return layer

  @staticmethod
  def trans_conv(inputs, filters, activation=tf.nn.relu, kernel_size=2, strides=2, l2_reg=None):
    """
    transposed convolution layer.

    Parameters
    ---------- 
      inputs: tf.Tensor
      filters: int 
        the number of the filter
      activation: 
        the activation function. The default function is the ReLu.
      kernel_size: int
        the kernel size. Default = 2
      strides: int
        strides. Default = 2
      l2_reg: None or float 
        the strengthen of the L2 regularization.

    Returns
    -------
      layer: tf.Tensor
    """
    regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg) if l2_reg is not None else None

    layer = tf.layers.conv2d_transpose(
      inputs=inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      kernel_regularizer=regularizer
    )

    return layer

  @staticmethod
  def pooling(inputs):
    return tf.layers.max_pooling2d(inputs=inputs, pool_size=2, strides=2)


  def UNet(self, is_training, l2_reg=None):
    """
    UNet structure.

    Parameters
    ----------
      l2_reg: None or float
        The strengthen of the L2 regularization.
      is_training: tf.bool
        Whether the session is for training or validation.

    Returns
    -------
      outputs: tf.Tensor
    """
    conv1_1 = self.conv2d(self.X, filters=64, l2_reg=l2_reg, is_training=is_training)
    conv1_2 = self.conv2d(conv1_1, filters=64, l2_reg=l2_reg, is_training=is_training)
    pool1 = self.pooling(conv1_2)

    conv2_1 = self.conv2d(pool1, filters=128, l2_reg=l2_reg, is_training=is_training)
    conv2_2 = self.conv2d(conv2_1, filters=128, l2_reg=l2_reg, is_training=is_training)
    pool2 = self.pooling(conv2_2)

    conv3_1 = self.conv2d(pool2, filters=256, l2_reg=l2_reg, is_training=is_training)
    conv3_2 = self.conv2d(conv3_1, filters=256, l2_reg=l2_reg, is_training=is_training)
    pool3 = self.pooling(conv3_2)

    conv4_1 = self.conv2d(pool3, filters=512, l2_reg=l2_reg, is_training=is_training)
    conv4_2 = self.conv2d(conv4_1, filters=512, l2_reg=l2_reg, is_training=is_training)
    pool4 = self.pooling(conv4_2)

    conv5_1 = self.conv2d(pool4, filters=1024, l2_reg=l2_reg)
    conv5_2 = self.conv2d(conv5_1, filters=1024, l2_reg=l2_reg)
    concat1 = tf.concat([conv4_2, self.trans_conv(conv5_2, filters=512, l2_reg=l2_reg)], axis=3)

    conv6_1 = self.conv2d(concat1, filters=512, l2_reg=l2_reg)
    conv6_2 = self.conv2d(conv6_1, filters=512, l2_reg=l2_reg)
    concat2 = tf.concat([conv3_2, self.trans_conv(conv6_2, filters=256, l2_reg=l2_reg)], axis=3)

    conv7_1 = self.conv2d(concat2, filters=256, l2_reg=l2_reg)
    conv7_2 = self.conv2d(conv7_1, filters=256, l2_reg=l2_reg)
    concat3 = tf.concat([conv2_2, self.trans_conv(conv7_2, filters=128, l2_reg=l2_reg)], axis=3)

    conv8_1 = self.conv2d(concat3, filters=128, l2_reg=l2_reg)
    conv8_2 = self.conv2d(conv8_1, filters=128, l2_reg=l2_reg)
    concat4 = tf.concat([conv1_2, self.trans_conv(conv8_2, filters=64, l2_reg=l2_reg)], axis=3)

    conv9_1 = self.conv2d(concat4, filters=64, l2_reg=l2_reg)
    conv9_2 = self.conv2d(conv9_1, filters=64, l2_reg=l2_reg)
    outputs = self.conv2d(conv9_2, filters=self.classes, kernel_size=1, activation=None)

    return outputs

  def train(self, parser):
    """
    training operation
    argument of this function are given by functions in main.py

    Parameters
    ----------
      parser: 
        the paser that has some options
    """
    epoch = parser.epoch
    l2 = parser.l2
    batch_size = parser.batch_size
    train_val_rate = parser.train_rate

    output = self.UNet(l2_reg=l2, is_training=self.is_training)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=output))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_ops = tf.train.AdamOptimizer(parser.learning_rate).minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=100)
    all_train, all_val = main.load_data(self.IMAGE_DIR, self.SEGMENTED_DIR, n_class=2, train_val_rate=train_val_rate)
    with tf.Session() as sess:
      init.run()
      for e in range(epoch):
        data = main.generate_data(*all_train, batch_size)
        val_data = main.generate_data(*all_val, len(all_val[0]))
        for Input, Teacher in data:
          sess.run(train_ops, feed_dict={self.X: Input, self.y: Teacher, self.is_training: True})
          ls = loss.eval(feed_dict={self.X: Input, self.y: Teacher, self.is_training: None})
          for val_Input, val_Teacher in val_data:
            val_loss = loss.eval(feed_dict={self.X: val_Input, self.y: val_Teacher, self.is_training: None})

        print(f'epoch #{e + 1}, loss = {ls}, val loss = {val_loss}')
        if e % 100 == 0:
          saver.save(sess, f"./params/model_{e + 1}epochs.ckpt")

      self.validation(sess, output)

  def validation(self, sess, output):
    val_image = main.load_data(self.VALIDATION_DIR, '', n_class=2, train_val_rate=1)[0]
    data = main.generate_data(*val_image, batch_size=1)
    for Input, _ in data:
      result = sess.run(output, feed_dict={self.X: Input, self.is_training: None}) 
      break
    
    result = np.argmax(result[0], axis=2)
    ident = np.identity(3, dtype=np.int8)
    result = ident[result]*255

    plt.imshow((Input[0]*255).astype(np.int16))
    plt.imshow(result, alpha=0.2)
    plt.show()
