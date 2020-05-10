import tensorflow as tf


class LatentMap(tf.keras.Model):
    def __init__(self, num_layer, latent_dim, hidden_dim):
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
        """
        Args:
            z: tf.Tensor, [B, z_dim]
        Returns:
            tf.Tensor, [B, latent_dim]
        """
        return self.blocks(z)
