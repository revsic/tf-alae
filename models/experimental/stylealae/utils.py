import numpy as np
import tensorflow as tf


class AffineTransform(tf.keras.Model):
    """Transform inputs in affine-method.
    """
    def __init__(self, shape):
        super(AffineTransform, self).__init__()
        self.shape = shape
        self.weight = tf.Variable(tf.random.normal(self.shape))
        self.bias = tf.Variable(tf.zeros(self.shape))

    def call(self, x):
        """Affine-transform the inputs.
        Args:
            x: tf.Tensor, arbitary shape.
        Returns:
            tf.Tensor, arbitary shape.
        """
        return x * self.weight + self.bias


class Normalize2D(tf.keras.Model):
    """Normalize inputs at height, width channels.
    """
    def __init__(self, eps=1e-8):
        super(Normalize2D, self).__init__()
        self.eps = eps

    def call(self, x):
        """Normalize inputs.
        Args:
            x: tf.Tensor, [B, H, W, C], 2D input tensor.
        Returns:
            tf.Tensor, [B, H, W, C], normalized tensor.
        """
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        return (x - mean) / (tf.math.sqrt(var) + self.eps)


class Repeat2D(tf.keras.Model):
    """Upsample 2D input tensor with repeating policy.
    """
    def __init__(self, factor):
        super(Repeat2D, self).__init__()
        self.factor = factor
    
    def call(self, x):
        """Upsample input tensor.
        Args:
            x: tf.Tensor, [B, H, W, C], input tensor.
        Returns:
            tf.Tensor, [B, H x factor, W x factor, C], upsampled.
        """
        _, h, w, c = x.shape
        x = x[:, :, None, :, None, :]
        x = tf.tile(x, [1, 1, self.factor, 1, self.factor, 1])
        return tf.reshape(x, [-1, h * self.factor, w * self.factor, c])


class Blur(tf.keras.Model):
    """Bluring image with predefined convolution kernel.
    """
    def __init__(self):
        super(Blur, self).__init__()
        # [3]
        kernel = np.array([1, 2, 1], dtype=np.float32)
        # [3, 3]
        kernel = kernel[:, None] * kernel[None, :]
        kernel /= kernel.sum()
        # [3, 3, 1, 1]
        self.kernel = tf.constant(kernel[..., None, None])
    
    def call(self, x):
        """Bluring image.
        Args:
            x: tf.Tensor, [B, H, W, C], image tensor.
        Returns:
            tf.Tensor, [B, H, W, C], blured image.
        """
        # C
        channel = tf.shape(x)[-1]
        # [3, 3, C, 1] = [kernel_height, kernel_width, in_channel, channel_mul]
        kernel = tf.tile(self.kernel, [1, 1, channel, 1])
        return tf.nn.depthwise_conv2d(
            x, kernel, strides=[1, 1, 1, 1], padding='SAME')
