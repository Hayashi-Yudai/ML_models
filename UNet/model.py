import tensorflow as tf
import numpy as np

def conv2d(
  inputs, filters, kernel_size=3, activation=tf.nn.relu, l2_reg=None, 
  momentum=0.95, epsilon=0.001, is_training=False,
  ):
  """
  convolutional layer. If the l2_reg is a float number, L2 regularization is imposed.
  Args:
    inputs: tf.Tensor
    filters: Non-zero positive integer. The number of the filter 
    activation: The activation function. The default is tf.nn.relu
    l2_reg: None or a float. The strengthen of the L2 regularization
    is_training: boolean (tf.Tensor). The default is False. If True, the batch normalization layer is added.
    momentum: float. The hyper parameter of the batch normalization layer
    epsilon: float. The hyper parameter of the batch normalization layer
  Returns:
    tf.Tensor
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

  if is_training:
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


def trans_conv(inputs, filters, activation=tf.nn.relu, kernel_size=2, strides=2, l2_reg=None):
  """
  transposed convolution layer.
  Args:
    inputs: input tensor
    filters: the number of the filter
    activation: the activation function. The default function is the ReLu.
    kernel_size: the kernel size. Default = 2
    strides: strides. Default = 2
    l2_reg: the strengthen of the L2 regularization. float or None
  Returns:
    tf.Tensor
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

def pooling(inputs):
  return tf.layers.max_pooling2d(inputs=inputs, pool_size=2, strides=2)


def UNet(inputs, classes, l2_reg=None, is_training=False):
  """
  UNet structure.
  Args:
    inputs: input images with (None, 128, 128, number of class) shape
    classes: the number of the class label
    l2_reg: float or None. The strengthen of the L2 regularization.
    is_training: boolean. Whether the session is for training or validation.
  Returns:
    tf.Tensor
  """
  conv1_1 = conv2d(inputs, filters=64, l2_reg=l2_reg, is_training=is_training)
  conv1_2 = conv2d(conv1_1, filters=64, l2_reg=l2_reg, is_training=is_training)
  pool1 = pooling(conv1_2)

  conv2_1 = conv2d(pool1, filters=128, l2_reg=l2_reg, is_training=is_training)
  conv2_2 = conv2d(conv2_1, filters=128, l2_reg=l2_reg, is_training=is_training)
  pool2 = pooling(conv2_2)

  conv3_1 = conv2d(pool2, filters=256, l2_reg=l2_reg, is_training=is_training)
  conv3_2 = conv2d(conv3_1, filters=256, l2_reg=l2_reg, is_training=is_training)
  pool3 = pooling(conv3_2)

  conv4_1 = conv2d(pool3, filters=512, l2_reg=l2_reg, is_training=is_training)
  conv4_2 = conv2d(conv4_1, filters=512, l2_reg=l2_reg, is_training=is_training)
  pool4 = pooling(conv4_2)

  conv5_1 = conv2d(pool4, filters=1024, l2_reg=l2_reg, is_training=is_training)
  conv5_2 = conv2d(conv5_1, filters=1024, l2_reg=l2_reg, is_training=is_training)
  concat1 = tf.concat([conv4_2, trans_conv(conv5_2, filters=512, l2_reg=l2_reg)], axis=3)

  conv6_1 = conv2d(concat1, filters=512, l2_reg=l2_reg, is_training=is_training)
  conv6_2 = conv2d(conv6_1, filters=512, l2_reg=l2_reg, is_training=is_training)
  concat2 = tf.concat([conv3_2, trans_conv(conv6_2, filters=256, l2_reg=l2_reg)], axis=3)

  conv7_1 = conv2d(concat2, filters=256, l2_reg=l2_reg, is_training=is_training)
  conv7_2 = conv2d(conv7_1, filters=256, l2_reg=l2_reg, is_training=is_training)
  concat3 = tf.concat([conv2_2, trans_conv(conv7_2, filters=128, l2_reg=l2_reg)], axis=3)

  conv8_1 = conv2d(concat3, filters=128, l2_reg=l2_reg, is_training=is_training)
  conv8_2 = conv2d(conv8_1, filters=128, l2_reg=l2_reg, is_training=is_training)
  concat4 = tf.concat([conv1_2, trans_conv(conv8_2, filters=64, l2_reg=l2_reg)], axis=3)

  conv9_1 = conv2d(concat4, filters=64, l2_reg=l2_reg, is_training=is_training)
  conv9_2 = conv2d(conv9_1, filters=64, l2_reg=l2_reg, is_training=is_training)
  outputs = conv2d(conv9_2, filters=classes, kernel_size=1, activation=None, is_training=is_training)

  return outputs

def train(data, parser):
  """
  training operation
  arguments of this function are given by functions in main.py
  Args:
    data: generator set of image and segmented image
    parser: the paser that has some options
  """
  epoch = parser.epoch
  batch_size = parser.batch_size

  X = tf.placeholder(tf.float32, [None, 128, 128, 3])
  y = tf.placeholder(tf.int32, [None, 128 , 128, 2])
  output = UNet(X, classes=2, l2_reg=parser.l2, is_training=True)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=output))
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  #TODO: introduce some metrics, accuracy... maybe

  with tf.control_dependencies(update_ops):
    train_ops = tf.train.AdamOptimizer(parser.learning_rate).minimize(loss)

  init = tf.global_variables_initializer()
  with tf.Session() as sess:
    init.run()
    for e in range(epoch):
      for Input, Teacher in data:
        #TODO: split training data and validation data
        sess.run(train_ops, feed_dict={X: [Input], y: [Teacher]})