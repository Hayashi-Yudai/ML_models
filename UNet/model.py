import tensorflow as tf
import numpy as np

def conv2d(inputs, filters, activation=tf.nn.relu):
  return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=3, padding='SAME', activation=activation)