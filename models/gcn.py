import tensorflow as tf
from typing import Tuple, List, Union

from .layers import GCNLayer
from .gcn_skip import GCNTwoLayersSkipConnection


class GCNSequential(tf.keras.Model):
    def __init__(self, layers: List[Union[GCNLayer,GCNTwoLayersSkipConnection]]):
        super(GCNSequential, self).__init__()
        self.model_layers = layers

    def call(self,inputs: Tuple[tf.SparseTensor,tf.Tensor],training=None):
        adj, nodes = inputs
        x = nodes
        for l in self.model_layers:
            if isinstance(l,(GCNLayer,GCNTwoLayersSkipConnection)):
                x = l([adj,x],training=training)
            else:
                x = l(x,training=training)
        return x
