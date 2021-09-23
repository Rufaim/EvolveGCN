import tensorflow as tf
from typing import Tuple



class GCNSkipLayer(tf.keras.layers.Layer):
    def __init__(self, units: int, activation=None, kernel_initializer="glorot_uniform", dtype=tf.float32):
        super(GCNSkipLayer, self).__init__(dtype=dtype)

        self.units = int(units)
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)

    def build(self, input_shape):
        last_dim_nodes = tf.TensorShape(input_shape[1])[-1]
        last_dim_skip = tf.TensorShape(input_shape[2])[-1]

        self.kernel_nodes = self.add_weight(
            'kernel_nodes',
            shape=[last_dim_nodes, self.units],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)
        self.kernel_skip = self.add_weight(
            'kernel_skip',
            shape=[last_dim_skip, self.units],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)
        self.built = True

    def call(self, inputs: Tuple[tf.SparseTensor,tf.Tensor,tf.Tensor], training=None, mask=None):
        adj, nodes, skip_input = inputs
        kernel_branch = tf.matmul(tf.sparse.sparse_dense_matmul(adj,nodes),self.kernel_nodes)
        skip_branch = tf.matmul(skip_input,self.kernel_skip)
        return self.activation(kernel_branch + skip_branch)