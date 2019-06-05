import tensorflow as tf
import numpy as np
from PIL import Image


def get_image():
  img = Image.open('./raw-img/train/butterfly/e030b20928e90021d85a5854ee454296eb70e3c818b413449df6c87ca3ed_640.jpg')
  img = img.resize([224, 224])
  img = np.asarray(img, dtype=np.float32)

  return img / 255.0

class ACoL:
  def __init__(self, class_num, threshold):
    self.class_num = class_num 
    self.threshold = threshold

    with tf.gfile.GFile('vgg16.pb', 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      _ = tf.import_graph_def(graph_def, name="")


  def ACoL(self):
      X = tf.get_default_graph().get_tensor_by_name('input_1:0')
      y = tf.get_default_graph().get_tensor_by_name('dense_2/Softmax:0')

      # VGG-16 backborn
      # block5_conv3
      block5_conv3 = tf.get_default_graph().get_tensor_by_name(
        'block5_conv3/Relu:0'
        )
      vgg_stop = tf.stop_gradient(block5_conv3)

      # First branch
      convA_1 = tf.layers.conv2d(
        inputs=block5_conv3,
        filters=1024,
        kernel_size=3,
        padding='SAME',
        activation=tf.nn.relu,
        name="convA_1"
        )
      convA_2 = tf.layers.conv2d(
        inputs=convA_1,
        filters=1024,
        kernel_size=3,
        padding='SAME',
        activation=tf.nn.relu
        )
      convA_3 = tf.layers.conv2d(
        inputs=convA_2,
        filters=self.class_num,
        kernel_size=1,
        padding='SAME',
        activation=tf.nn.relu
      )
      global_ave_pool_A = tf.reduce_mean(convA_3, axis=[1, 2])

      # Second branch
      feature_map_A = convA_3[:, :, :, tf.math.argmax(global_ave_pool_A[0])]
      th = tf.constant(self.threshold, shape=[14, 14], dtype=tf.float32)
      feature_map_A = tf.maximum(feature_map_A, th)

      return block5_conv3 - feature_map_A
      """
      convB_1 = tf.layers.conv2d(
        inputs=block5_conv3 - feature_map_A,
        filters=1024,
        kernel_size=3,
        padding='SAME',
        activation=tf.nn.relu
        )
      convB_2 = tf.layers.conv2d(
        inputs=convA_1,
        filters=1024,
        kernel_size=3,
        padding='SAME',
        activation=tf.nn.relu
        )
      convB_3 = tf.layers.conv2d(
        inputs=convA_2,
        filters=self.class_num,
        kernel_size=1,
        padding='SAME',
        activation=tf.nn.relu
      )
      global_ave_pool_B = tf.reduce_mean(convB_3, axis=[1, 2])

      feature_map_B =convB_3[:, :, :, tf.math.argmax(global_ave_pool_A)]
      th = tf.constant(self.threshold, shape=[14, 14], dtype=tf.float32)
      feature_map_B = tf.maximum(feature_map_B, th)

      return tf.maximum(feature_map_A, feature_map_B)
      """


  def train(self):

    X = tf.get_default_graph().get_tensor_by_name('input_1:0')
    y = tf.get_default_graph().get_tensor_by_name('dense_2/Softmax:0')

    with tf.Session() as sess:
      fw = tf.summary.FileWriter('./', sess.graph)
      res = sess.run(y, feed_dict={X: [get_image()]})

    fw.close()

if __name__ == '__main__':
    with tf.gfile.GFile('vgg16.pb', 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      _ = tf.import_graph_def(graph_def, name="")

    X = tf.get_default_graph().get_tensor_by_name('input_1:0')
    model = ACoL(10, 0.9)
    test = model.ACoL()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
      init.run()
      res = sess.run(test, feed_dict={X: [get_image()]})

    print(res.shape)