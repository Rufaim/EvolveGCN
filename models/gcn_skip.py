import tensorflow as tf
from typing import Tuple, Union

from .layers import GCNLayer, GCNSkipLayer



class GCNTwoLayersSkipConnection(tf.keras.Model):
    def __init__(self, layer_gcn: GCNLayer, layer_gcn_skip: GCNSkipLayer, dropout: Union[tf.keras.layers.Dropout,None]):
        super(GCNTwoLayersSkipConnection, self).__init__()
        self.layer_gcn = layer_gcn
        self.dropout = dropout
        self.layer_gcn_skip = layer_gcn_skip

    def call(self, inputs: Tuple[tf.SparseTensor, tf.Tensor], training=None):
        adj, nodes = inputs
        x = self.layer_gcn([adj,nodes])
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        x = self.layer_gcn_skip([adj,x,nodes])
        return x
