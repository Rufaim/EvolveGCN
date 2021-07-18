import tensorflow as tf
from typing import Tuple


class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, units: int, activation=None, kernel_initializer="glorot_uniform", dtype=tf.float32):
        super(GCNLayer, self).__init__(dtype=dtype)

        self.units = int(units)
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)


    def build(self, input_shape):
        last_dim = tf.TensorShape(input_shape[1])[-1]

        self.kernel = self.add_weight(
            'kernel',
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)
        self.built = True

    def call(self, inputs: Tuple[tf.SparseTensor,tf.Tensor], training=None, mask=None):
        adj, nodes = inputs
        return tf.matmul(tf.sparse.sparse_dense_matmul(adj,nodes),self.kernel)
