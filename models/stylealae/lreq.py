import numpy as np
import tensorflow as tf


def make_tuple(x):
    if isinstance(x, int):
        return (x, x)
    return x


class LrEqDense(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 gain=np.sqrt(2.),
                 lrmul=1.0):
        super(LrEqDense, self).__init__()
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.gain = gain
        self.lrmul = lrmul
    
    def build(self, input_shape):
        features = input_shape[-1]
        self.std = self.gain / np.sqrt(features)
        
        kernel_init = tf.random_normal_initializer(0, self.std)
        self.kernel = tf.Variable(
            kernel_init([input_shape[-1], self.units]),
            trainable=True)
        self.kernel.lreq_coeff = self.std * self.lrmul

        if self.use_bias:
            bias_init = tf.zeros_initializer()
            self.bias = tf.Variable(
                bias_init([self.units]), trainable=True)
            self.bias.lreq_coeff = self.lrmul
    
    def call(self, inputs):
        x = tf.matmul(inputs, self.kernel)
        if self.use_bias:
            x = x + self.bias
        if self.activation is not None:
            x = self.activation(x)
        return x


class LrEqConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='VALID',
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 gain=np.sqrt(2.),
                 lrmul=1.0,
                 transform_kernel=False):
        super(LrEqConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = make_tuple(kernel_size)
        self.strides = make_tuple(strides)
        self.padding = padding
        self.dilation_rate = make_tuple(dilation_rate)
        self.activation = activation
        self.use_bias = use_bias
        self.gain = gain
        self.lrmul = lrmul
        self.transform_kernel = transform_kernel
    
    def build(self, input_shape):
        features = input_shape[-1]
        self.std = self.gain / np.sqrt(np.prod(self.kernel_size) * features)

        kernel_init = tf.random_normal_initializer(0, self.std)
        self.kernel = tf.Variable(
            kernel_init([*self.kernel_size, features, self.filters]),
            trainable=True)
        self.kernel.lreq_coeff = self.std * self.lrmul

        if self.use_bias:
            bias_init = tf.zeros_initializer()
            self.bias = tf.Variable(
                bias_init([self.filters]), trainable=True)
            self.bias.lreq_coeff = self.lrmul
    
    def call(self, inputs):
        kernel = self.kernel
        if self.transform_kernel:
            kernel = tf.pad(
                kernel, [[1, 1], [1, 1], [0, 0], [0, 0]], 'constant')
            kernel = 0.25 * (
                kernel[1:, 1:] + 
                kernel[:-1, 1:] + 
                kernel[1:, :-1] + 
                kernel[:-1, :-1])

        x = tf.nn.conv2d(inputs,
                         kernel,
                         self.strides,
                         self.padding,
                         dilations=self.dilation_rate)
        if self.use_bias:
            x = x + self.bias
        if self.activation is not None:
            x = self.activation(x)
        return x


class LrEqConv2DTranspose(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='VALID',
                 output_padding=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 gain=np.sqrt(2.),
                 lrmul=1.0,
                 transform_kernel=False):
        super(LrEqConv2DTranspose, self).__init__()
        self.filters = filters
        self.kernel_size = make_tuple(kernel_size)
        self.strides = make_tuple(strides)
        self.padding = padding
        self.output_padding = output_padding
        self.dilation_rate = make_tuple(dilation_rate)
        self.activation = activation
        self.use_bias = use_bias
        self.gain = gain
        self.lrmul = lrmul
        self.transform_kernel = transform_kernel
    
    def build(self, input_shape):
        features = input_shape[-1]
        self.std = self.gain / np.sqrt(np.prod(self.kernel_size) * features)

        if self.padding == 'VALID':
            raise NotImplementedError('"VALID" padding is not supported')
        elif self.padding == 'SAME':
            out_shape = [input_shape[i + 1] * self.strides[i] for i in range(2)]
        else:
            raise ValueError('padding should be one of "VALID" or "SAME"')
        self.out_shape = [*out_shape, self.filters]

        kernel_init = tf.random_normal_initializer(0, self.std)
        self.kernel = tf.Variable(
            kernel_init([*self.kernel_size, self.filters, features]),
            trainable=True)
        self.kernel.lreq_coeff = self.std * self.lrmul

        if self.use_bias:
            bias_init = tf.zeros_initializer()
            self.bias = tf.Variable(
                bias_init([self.filters]), trainable=True)
            self.bias.lreq_coeff = self.lrmul
    
    def call(self, inputs):
        kernel = self.kernel
        if self.transform_kernel:
            kernel = tf.pad(
                kernel, [[1, 1], [1, 1], [0, 0], [0, 0]], 'constant')
            kernel = \
                kernel[1:, 1:] + \
                kernel[:-1, 1:] + \
                kernel[1:, :-1] + \
                kernel[:-1, :-1]

        x = tf.nn.conv2d_transpose(
            inputs,
            kernel,
            [tf.shape(inputs)[0], *self.out_shape],
            self.strides,
            self.padding,
            dilations=self.dilation_rate)
        if self.use_bias:
            x = x + self.bias
        if self.activation is not None:
            x = self.activation(x)
        return x
