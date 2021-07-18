import tensorflow as tf
from typing import Tuple
from .h_gru import HGRUCell
from .summarize import SummarizeLayer



class EGCUH(tf.keras.Model):
    def __init__(self, gru_cell: HGRUCell, summarize: SummarizeLayer, activation=None, dtype=tf.float32):
        super(EGCUH, self).__init__(dtype=dtype)

        self.gru_cell = gru_cell
        self.activation = tf.keras.activations.get(activation)
        self.summarize = summarize

    def call(self, inputs: Tuple[tf.SparseTensor,tf.Tensor,tf.Tensor], training=None, mask=None) -> Tuple[tf.Tensor,tf.Tensor]:
        adj,nodes,weigths = inputs
        node_summary = self.summarize([nodes,tf.shape(weigths)[-1]])
        weigths_new = self.gru_cell([tf.transpose(node_summary),weigths])

        nodes_new = self.activation(tf.matmul(tf.sparse.sparse_dense_matmul(adj,nodes),weigths_new))
        return nodes_new, weigths_new

    def get_initial_weigths(self, input_shape) -> tf.Tensor:
        return self.gru_cell.get_initial_state(input_shape)