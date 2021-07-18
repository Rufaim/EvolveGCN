import tensorflow as tf
from typing import List, Tuple
from .layers import EGCUH



class EvolveGCN(tf.keras.Model):
    def __init__(self, layers: List[EGCUH]):
        super(EvolveGCN, self).__init__()
        self.layers_ = layers

    def call(self, inputs: Tuple[tf.SparseTensor,tf.Tensor,List[tf.Tensor]], training=None, mask=None):
        adj, nodes, weigts = inputs
        new_weigths = []
        for i in range(len(self.layers_)):
            nodes,nw = self.layers_[i]([adj,nodes,weigts[i]])
            new_weigths.append(nw)
        return nodes, new_weigths

    def get_initial_weigths(self, input_shape) -> List[tf.Tensor]:
        states = []
        s = input_shape
        for l in self.layers_:
            s = l.get_initial_weigths(s)
            states.append(s)
            s = tf.shape(s)
        return states
