import tensorflow as tf

from ..alae import ALAE
from .encoder import Encoder
from .generator import Generator
from .maplatent import LatentMap


class StyleAlae(ALAE):
    def __init__(self, settings):
        super(ALAE, self).__init__()
        self.settings = settings
        self.prepare(self.settings['z_dim'],
                     self.settings['gamma'],
                     self.settings['learning_rate'],
                     self.settings['beta1'],
                     self.settings['beta2'])

        self.binder = Binder(self.enc, self.disc)
        self.fakepass = tf.keras.Sequential([self.map, self.gen, self.binder])
        self.realpass = self.binder
        self.latentpass = self.encode

    def encode(self, *args, **kwargs):
        """Encode the input tensors to latent vectors.
        Args:
            _: tf.Tensor, [B, ...], input tensors.
        Returns:
            _: tf.Tensor, [B, latent_dim], latent vectors.
        """
        x, style = self.enc(*args, **kwargs)
        return style

    def mapper(self):
        """Model for mpping latent from prior.
        Returns:
            tf.keras.Model: map prior to latent.
        """
        return LatentMap(num_layer=self.settings['map_num_layers'],
                         z_dim=self.settings['latent_dim'],
                         latent_dim=self.settings['latent_dim'],
                         hidden_dim=self.settings['latent_dim'])

    def generator(self):
        """Model for generating data from encoded latent.
        Returns:
            tf.keras.Model: generate sample from encoded latent.
        """
        return Generator(init_channels=self.settings['init_channels'],
                         max_channels=self.settings['max_channels'],
                         num_layer=self.settings['num_layers'],
                         out_channels=self.settings['out_channels'])

    def encoder(self):
        """Model for encoding data to fixed length latent vector.
        Returns:
            tf.keras.Model: encode data to fixed length latent vector.
        """
        return Encoder(init_channels=self.settings['init_channels'],
                       max_channels=self.settings['max_channels'],
                       num_layer=self.settings['num_layers'],
                       latent_dim=self.settings['latent_dim'])

    def discriminator(self):
        """Model for discriminating real sample from fake one.
        Returns:
            tf.keras.Model: discriminate real sample from fake one.
        """
        return tf.keras.layers.Dense(1)

    @staticmethod
    def default_setting():
        return {
            'latent_dim': 256,
            'num_layers': 3,
            'map_num_layers': 5,
            'init_channels': 32,
            'max_channels': 256,
            'out_channels': 3,
            'lr': 0.002,
            'beta1': 0.0,
            'beta2': 0.99,
            'gamma': 10,
        }

    class Binder(tf.keras.Model):
        def __init__(self, encoder, discriminator):
            super(Binder, self).__init__()
            self.encoder = encoder
            self.discriminator = discriminator
        
        def call(self, x):
            x, style = self.encoder(x)
            return self.discriminator(x)
