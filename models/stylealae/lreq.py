import numpy as np
import tensorflow as tf


def make_tuple(x):
    """Make inputs to tuple.
    Args:
        x: Union[int, Tuple[int, int]], inputs.
    Returns:
        Tuple[int, int], tuple.
    """
    if isinstance(x, int):
        return (x, x)
    return x


class LrEqDense(tf.keras.layers.Layer):
    """Learning rate equalized dense layer.
    """
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 gain=np.sqrt(2.),
                 lrmul=1.0):
        """Initializer.
        Args:
            units: int, the number of the output features.
            activation: Optional[Callable[[tf.Tensor], tf.Tensor]],
                optional activation function.
            use_bias: bool, whether use bias or not.
            gain: float, stddev gain for kernel initialization.
            lrmul: float, multiplicative factor of learning rates for lreq optimizer.
        """
        super(LrEqDense, self).__init__()
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.gain = gain
        self.lrmul = lrmul
    
    def build(self, input_shape):
        """Build module, initialize kernel and bias.
        Args:
            input_shape: List[int], shape of the input.
        """
        features = input_shape[-1]
        self.std = self.gain / np.sqrt(features)
        # initialize kernel
        kernel_init = tf.random_normal_initializer(0, self.std)
        self.kernel = tf.Variable(
            kernel_init([input_shape[-1], self.units]),
            trainable=True)
        # set lreq coefficient
        self.kernel.lreq_coeff = self.std * self.lrmul

        if self.use_bias:
            # initialize bias
            bias_init = tf.zeros_initializer()
            self.bias = tf.Variable(
                bias_init([self.units]), trainable=True)
            self.bias.lreq_coeff = self.lrmul
    
    def call(self, inputs):
        """Linear projection of given inputs.
        Args:
            inputs: tf.Tensor, [..., C], input tensor.
        Returns:
            tf.Tensor, [..., units], projected.
        """
        x = tf.matmul(inputs, self.kernel)
        if self.use_bias:
            x = x + self.bias
        if self.activation is not None:
            x = self.activation(x)
        return x


class LrEqConv2D(tf.keras.layers.Layer):
    """Learning rate equalized 2d convolution.
    """
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
        """Initializer.
        Args:
            filters: int, the number of the output channels.
            kernel_size: Union[int, Tuple[int, int]], size of the kernel.
            strides: Union[int, Tuple[int, int]], strides.
            padding: str, padding policy, VALID or SAME.
            dilation_rate: Union[int, Tuple[int, int]], dilation rate.
            activation: Optional[Callable[[tf.Tensor], tf.Tensor]],
                optional activation function.
            use_bias: bool, whether use bias or not.
            gain: float, stddev gain for kernel initialization.
            lrmul: float, multiplicative factor of learning rates for lreq optimizer.
            transform_kernel: bool, whether transform kernel or not.
        """
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
        """Build module, initialize kernel and bias.
        Args:
            input_shape: List[int], shape of the input.
        """
        features = input_shape[-1]
        self.std = self.gain / np.sqrt(np.prod(self.kernel_size) * features)
        # initialize kernel
        kernel_init = tf.random_normal_initializer(0, self.std)
        self.kernel = tf.Variable(
            kernel_init([*self.kernel_size, features, self.filters]),
            trainable=True)
        # set lreq coefficient
        self.kernel.lreq_coeff = self.std * self.lrmul

        if self.use_bias:
            # initialize bias
            bias_init = tf.zeros_initializer()
            self.bias = tf.Variable(
                bias_init([self.filters]), trainable=True)
            self.bias.lreq_coeff = self.lrmul
    
    def call(self, inputs):
        """Run 2d convolution.
        Args:
            inputs: tf.Tensor, [B, H, W, C], input tensor.
        Returns:
            tf.Tensor, [B, H', W', filters], output tensor.
        """
        kernel = self.kernel
        if self.transform_kernel:
            # [kernel_height + 2, kernel_width + 2, C, filters]
            kernel = tf.pad(
                kernel, [[1, 1], [1, 1], [0, 0], [0, 0]], 'constant')
            # bluring kernel
            # [kernel_height + 1, kernel_width + 1, C, filters]
            kernel = 0.25 * (
                kernel[1:, 1:] + 
                kernel[:-1, 1:] + 
                kernel[1:, :-1] + 
                kernel[:-1, :-1])
        # [B, H', W', filters]
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
    """Learning rate equalized transposed 2d convolution.
    """
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
        """Initializer.
        Args:
            filters: int, the number of the output channels.
            kernel_size: Union[int, Tuple[int, int]], size of the kernel.
            strides: Union[int, Tuple[int, int]], strides.
            padding: str, padding policy, VALID or SAME.
            dilation_rate: Union[int, Tuple[int, int]], dilation rate.
            activation: Optional[Callable[[tf.Tensor], tf.Tensor]],
                optional activation function.
            use_bias: bool, whether use bias or not.
            gain: float, stddev gain for kernel initialization.
            lrmul: float, multiplicative factor of learning rates for lreq optimizer.
            transform_kernel: bool, whether transform kernel or not.
        """
        super(LrEqConv2DTranspose, self).__init__()
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
        """Build module, initialize kernel and bias.
        Args:
            input_shape: List[int], shape of the input.
        """
        features = input_shape[-1]
        self.std = self.gain / np.sqrt(np.prod(self.kernel_size) * features)

        # currently, valid padding is not supported due to output shape computation
        if self.padding == 'VALID':
            raise NotImplementedError('"VALID" padding is not supported')
        elif self.padding == 'SAME':
            out_shape = [input_shape[i + 1] * self.strides[i] for i in range(2)]
        else:
            raise ValueError('padding should be one of "VALID" or "SAME"')
        self.out_shape = [*out_shape, self.filters]
        # initialize kernel
        kernel_init = tf.random_normal_initializer(0, self.std)
        self.kernel = tf.Variable(
            kernel_init([*self.kernel_size, self.filters, features]),
            trainable=True)
        self.kernel.lreq_coeff = self.std * self.lrmul

        if self.use_bias:
            # initialize bias
            bias_init = tf.zeros_initializer()
            self.bias = tf.Variable(
                bias_init([self.filters]), trainable=True)
            self.bias.lreq_coeff = self.lrmul
    
    def call(self, inputs):
        """Run transposed 2d convolution.
        Args:
            inputs: tf.Tensor, [B, H, W, C], input tensor.
        Returns:
            tf.Tensor, [B, H', W', filters], output tensor.
        """
        kernel = self.kernel
        if self.transform_kernel:
            # [kernel_height + 2, kernel_width + 2, filters, C]
            kernel = tf.pad(
                kernel, [[1, 1], [1, 1], [0, 0], [0, 0]], 'constant')
            # [kernel_height + 1, kernel_width + 1, filters, C]
            kernel = \
                kernel[1:, 1:] + \
                kernel[:-1, 1:] + \
                kernel[1:, :-1] + \
                kernel[:-1, :-1]
        # [B, H', W', filters]
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
