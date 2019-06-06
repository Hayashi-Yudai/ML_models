import tensorflow as tf
import numpy as np
from PIL import Image

saver = tf.train.import_meta_graph('./vgg16.ckpt.meta')

with tf.name_scope('io'):
  X = tf.get_default_graph().get_tensor_by_name('inputs:0')

with tf.name_scope('vgg16'):
  conv5_3 = tf.get_default_graph().get_tensor_by_name('vgg16/conv5/3/Relu:0')

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
  flatten = tf.reduce_mean(conv3, axis=[1, 2])

# For test only
img = Image.open('/home/yudai/Pictures/raw-img/train/cat/936.jpeg')
img = img.resize([224, 224])
img = np.asarray(img, dtype=np.float32)
img /= 255.0

init = tf.global_variables_initializer()
with tf.Session() as sess:
  init.run()
  saver.restore(sess, './vgg16.ckpt')
  
  res = sess.run(flatten, feed_dict={X: [img]})

print(res)