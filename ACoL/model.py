import tensorflow as tf
import tflearn
import numpy as np
from PIL import Image

saver = tf.train.import_meta_graph('./vgg16/vgg16.ckpt.meta')

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
  flatten = tflearn.layers.conv.global_avg_pool(conv3)
  M = tf.math.argmax(flatten, axis=1)
  max_list = []
  for i in range(2):
    max_list.append(conv3[i, M[i]])

  feature_map_A = tf.stack(max_list)

with tf.name_scope('branch_B'):
  threshold = 0.9


# For test only
img = Image.open('/home/yudai/Pictures/raw-img/train/cat/936.jpeg')
img = img.resize([224, 224])
img = np.asarray(img, dtype=np.float32)
img /= 255.0
img2 = Image.open('/home/yudai/Pictures/raw-img/train/cat/41.jpeg')
img2 = img2.resize([224, 224])
img2 = np.asarray(img2, dtype=np.float32)
img2 /= 255.0


init = tf.global_variables_initializer()
with tf.Session() as sess:
  init.run()
  saver.restore(sess, './vgg16/vgg16.ckpt')
  res = sess.run(feature_map_A, feed_dict={X: [img, img2]})

print(res)