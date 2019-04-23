import tensorflow as tf
import numpy as np

def conv2d(
  inputs, filters, activation=tf.nn.relu, l2_reg=None, is_training=False,
  momentum=0.95, epsilon=0.001
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
    kernel_size=3,
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
  regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg) if l2_reg is not None else None

  layer = tf.layers.conv2d_transpose(
    inputs=inputs,
    filters=filters,
    kernel_size=kernel_size,
    strides=strides,
    kernel_regularizer=regularizer
  )

  return layer