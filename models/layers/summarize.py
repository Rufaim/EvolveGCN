import tensorflow as tf
from typing import Tuple



class SummarizeLayer(tf.keras.layers.Layer):
    def __init__(self,kernel_initializer="glorot_uniform", dtype=tf.float32):
        super(SummarizeLayer, self).__init__(dtype=dtype)
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        last_dim_inp = tf.TensorShape(input_shape[0])[-1]

        self.p = self.add_weight(
            'p',
            shape=[last_dim_inp],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)

        self.built = True

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training=None, mask=None):
        x,k = inputs
        y = tf.linalg.matvec(x,self.p) / tf.linalg.norm(self.p)
        top_y = tf.math.top_k(y,k)
        out = tf.gather(x,top_y.indices,axis=0) * tf.expand_dims(tf.tanh(top_y.values),-1)
        return out
