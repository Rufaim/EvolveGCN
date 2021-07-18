import tensorflow as tf
from typing import Tuple, List

from .layers import GCNLayer



class GCNSequential(tf.keras.Model):
    def __init__(self, layers: List[GCNLayer]):
        super(GCNSequential, self).__init__()
        self.model_layers = layers

    def call(self,inputs: Tuple[tf.SparseTensor,tf.Tensor],training=None):
        adj, nodes = inputs
        x = nodes
        for l in self.model_layers:
            if isinstance(l,GCNLayer):
                x = l([adj,x],training=training)
            else:
                x = l(x,training=training)
        return x
