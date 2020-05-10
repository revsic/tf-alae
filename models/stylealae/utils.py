import tensorflow as tf


class AffineTransform(tf.keras.Model):
    def __init__(self, shape):
        super(AffineTransform, self).__init__()
        self.shape = shape
        self.weight = tf.Variable(tf.random.normal(self.shape))
        self.bias = tf.Variable(tf.zeros(self.shape))

    def call(self, x):
        """
        Args:
            x: tf.Tensor, arbitary shape
        Returns:
            tf.Tensor, arbitary shape
        """
        return x * self.weight + self.bias


class Normalize2D(tf.keras.Model):
    def __init__(self, eps=1e-8):
        super(Normalize2D, self).__init__()
        self.eps = eps

    def call(self, x):
        """
        Args:
            x: tf.Tensor, [B, H, W, C]
        Returns:
            tf.Tensor, [B, H, W, C]
        """
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        return (x - mean) / (tf.math.sqrt(var) + self.eps)


class Repeat2D(tf.keras.Model):
    def __init__(self, factor):
        super(Repeat2D, self).__init__()
        self.factor = factor
    
    def call(self, x):
        """
        Args:
            x: tf.Tensor, [B, H, W, C]
        Returns:
            tf.Tensor, [B, H x factor, W x factor, C]
        """
        _, h, w, c = x.shape
        x = x[:, :, None, :, None, :]
        x = tf.tile(x, [1, 1, self.factor, 1, self.factor, 1])
        return tf.reshape(x, [-1, h * self.factor, w * self.factor, c])
