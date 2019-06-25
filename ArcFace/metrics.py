import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

class ArcFace(Layer):
    def __init__(self, n_classes=1000, s=30.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = tf.keras.regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name="W", 
                                 shape=(512, self.n_classes),
                                 initializer="glorot_uniform",
                                 trainable=True,
                                 regularizer=self.regularizer
                                )
    
    def call(self, inputs):
        x, y = inputs
        x = tf.nn.l2_normalize(x, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)
        logits = x @ W

        theta = tf.acos(K.clip(logits, -1.0+K.epsilon(), 1.0-K.epsilon()))
        target_logits = tf.cos(theta + self.m)
        
        logits = logits * (1-y) + target_logits * y
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)