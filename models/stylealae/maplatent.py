import tensorflow as tf

from .lreq import LrEqDense


class LatentMap(tf.keras.Model):
    """Map prior to latent or latent to prior.
    """
    def __init__(self, num_layer, latent_dim, hidden_dim):
        """Initializer.
        Args:
            num_layer: int, number of the layer.
            latent_dim: int, size of the style vector.
            hidden_dim: int, size of the hidden units.
        """
        super(LatentMap, self).__init__()
        self.num_layer = num_layer
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.leaky_relu = tf.keras.layers.LeakyReLU(0.2)
        self.blocks = tf.keras.Sequential([
            LrEqDense(self.hidden_dim, activation=self.leaky_relu, lrmul=0.1)
            for _ in range(self.num_layer - 1)])

        self.blocks.add(LrEqDense(self.latent_dim, lrmul=0.1))

    def call(self, z):
        """Fully-connected pass.
        Args:
            z: tf.Tensor, [B, z_dim], style prior.
        Returns:
            tf.Tensor, [B, latent_dim], style vector.
        """
        return self.blocks(z)
