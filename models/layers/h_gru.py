import tensorflow as tf
from typing import Tuple



class HGRUCell(tf.keras.layers.Layer):
    def __init__(self, units: int, activation='tanh',
                    recurrent_activation='sigmoid',
                    use_bias=True,
                    kernel_initializer='glorot_uniform',
                    recurrent_initializer='orthogonal',
                    bias_initializer='zeros',
                    dtype=tf.float32):
        super(HGRUCell, self).__init__(dtype=dtype)

        self.units = int(units)
        self.activation = tf.keras.activations.get(activation)
        self.recurrent_activation = tf.keras.activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.recurrent_initializer = tf.keras.initializers.get(recurrent_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

    def build(self, input_shape):
        inp_shape = tf.TensorShape(input_shape[0])
        rec_shape = tf.TensorShape(input_shape[1])
        last_dim_inp = inp_shape[-1]
        last_dim_rec = rec_shape[-1]

        self.kernel_inp_x = self.add_weight(
            'kernel_input_x',
            shape=[last_dim_inp, 2*self.units],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)
        self.kernel_inp_h = self.add_weight(
            'kernel_input_h',
            shape=[last_dim_rec, 2*self.units],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)

        self.kernel_rec_x = self.add_weight(
            "kernel_recurrent_x",
            shape=[last_dim_inp, self.units],
            initializer=self.recurrent_initializer,
            dtype=self.dtype,
            trainable=True)
        self.kernel_rec_h = self.add_weight(
            "kernel_recurrent_h",
            shape=[last_dim_rec, self.units],
            initializer=self.recurrent_initializer,
            dtype=self.dtype,
            trainable=True)

        if self.use_bias:
            self.bias_inp = self.add_weight(
                "bias_input",
                shape=[1, 2*self.units],
                initializer=self.bias_initializer,
                dtype=self.dtype,
                trainable=True)
            self.bias_rec = self.add_weight(
                "bias_recurrent",
                shape=[1, self.units],
                initializer=self.bias_initializer,
                dtype=self.dtype,
                trainable=True)

        self.built = True

    def call(self, inputs: Tuple[tf.Tensor,tf.Tensor], training=None, mask=None):
        X,H = inputs
        ZR = self.activation(tf.matmul(X,self.kernel_inp_x) + tf.matmul(H, self.kernel_inp_h) + self.bias_inp)
        Z, R = tf.split(ZR,2,axis=-1)
        H_new = self.recurrent_activation(tf.matmul(X,self.kernel_rec_x) + tf.matmul(R*H, self.kernel_rec_h) + self.bias_rec)
        H_new = (1 - Z) * H + Z * H_new
        return H_new

    def get_initial_state(self, input_shape) -> tf.Tensor:
        inp_shape = tf.TensorShape(input_shape)
        return tf.zeros(inp_shape[-1:] + [self.units])
