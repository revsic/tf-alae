import tensorflow as tf


class LatentMap(tf.keras.Model):
    """Generate style vector from prior.
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
            tf.keras.layers.Dense(self.hidden_dim, activation=self.leaky_relu)
            for _ in range(self.num_layer - 1)])

        self.blocks.add(
            tf.keras.layers.Dense(self.latent_dim))

    def call(self, z):
        """Generate style vector.
        Args:
            z: tf.Tensor, [B, z_dim], style prior.
        Returns:
            tf.Tensor, [B, latent_dim], style vector.
        """
        return self.blocks(z)
