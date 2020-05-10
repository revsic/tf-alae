import tensorflow as tf

from .alae import ALAE


class MlpAlae(ALAE):
    """ALAE with multi-layer perceptron architecture.
    """
    def __init__(self, settings=None):
        """Constructor.
        Args:
            settings: Dict[str, any], model settings,
                reference `MlpAlae.default_setting`.
        """
        super(MlpAlae, self).__init__()
        if settings is None:
            settings = self.default_setting()
        
        self.settings = settings
        self.z_dim = settings['z_dim']
        self.latent_dim = settings['latent_dim']
        self.output_dim = settings['output_dim']

        self.prepare(self.z_dim,
                     settings['gamma'],
                     settings['lr'],
                     settings['beta1'],
                     settings['beta2'])

    def _seq(self, dims, input_dim, activation=tf.nn.relu):
        layer = tf.keras.Sequential([
            tf.keras.layers.Dense(dim, activation=activation)
            for dim in dims[:-1]])
        layer.add(
            tf.keras.layers.Dense(dims[-1]))
        layer.build((None, input_dim))
        return layer

    def mapper(self, *args, **kwargs):
        """Model for mapping latent from prior.
        Returns:
            tf.keras.Model: tf.Tensor[B, z_dim] -> tf.Tensor[B, latent_dim],
                map prior to latent.
        """
        return self._seq(self.settings['f'], self.z_dim)

    def generator(self, *args, **kwargs):
        """Model for generating data from encoded latent.
        Returns:
            tf.keras.Model: tf.Tensor[B, latent_dim] -> tf.Tensor[B, output_dim],
                generate sample from encoded latent.
        """
        return self._seq(self.settings['g'], self.latent_dim)

    def encoder(self, *args, **kwargs):
        """Model for encoding data to fixed length latent vector.
        Returns:
            tf.keras.Model: tf.Tensor[B, output_dim] -> tf.Tensor[B, latent_dim]
                encode data to fixed length latent vector.
        """
        return self._seq(self.settings['e'], self.output_dim)

    def discriminator(self, *args, **kwargs):
        """Model for discriminating real sample from fake one.
        Returns:
            tf.keras.Model: tf.Tensor[B, latent_dim] -> tf.Tensor[B, 1]
                discriminate real sample from fake one.
        """
        return self._seq(self.settings['d'], self.latent_dim)

    @staticmethod
    def default_setting(z_dim=128, latent_dim=50, output_dim=784 + 10):
        """Default settings.
        """
        return {
            'z_dim': z_dim,
            'latent_dim': latent_dim,
            'output_dim': output_dim,
            'gamma': 10,
            'f': [1024, latent_dim],
            'g': [1024, output_dim],
            'e': [1024, latent_dim],
            'd': [1024, 1],
            'lr': 0.002,
            'beta1': 0.0,
            'beta2': 0.99,
        }
