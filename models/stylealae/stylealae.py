import tensorflow as tf

from ..alae import ALAE
from .encoder import Encoder
from .generator import Generator
from .maplatent import LatentMap


class StyleAlae(ALAE):
    """Style-ALAE, Adversarial latent autoencoder with StyleGAN architecture.
    """
    def __init__(self, settings=None):
        """Initializer.
        Args:
            settings: Dict[str, Any], parameters for constructing architecture.
        """
        super(ALAE, self).__init__()
        if settings is None:
            settings = StyleAlae.default_setting()

        self.settings = settings
        self.latent_dim = self.settings['latent_dim']
        self.num_layer = self.settings['num_layers']
        self.map_num_layer = self.settings['map_num_layers']
        self.init_channels = self.settings['init_channels']
        self.max_channels = self.settings['max_channels']
        self.out_channels = self.settings['out_channels']

        self.img_size = 2 ** (self.num_layer + 1)

        self.prepare(self.latent_dim,
                     self.settings['gamma'],
                     self.settings['lr'],
                     self.settings['beta1'],
                     self.settings['beta2'])

        self.binder = StyleAlae.Binder(self.enc, self.disc)
        self.styleonly = StyleAlae.StyleOnly(self.enc)

        self.fakepass = tf.keras.Sequential([self.map, self.gen, self.binder])
        self.realpass = self.binder
        self.latentpass = tf.keras.Sequential([
            self.map, self.gen, self.styleonly])
        
        style_variables = []
        for block in self.enc.blocks:
            style_variables += [
                x.name for x in block.style_proj1.trainable_variables]
            style_variables += [
                x.name for x in block.style_proj2.trainable_variables]

        self.ed_var = [
            var for var in self.ed_var if var.name not in style_variables]

    def encode(self, *args, **kwargs):
        """Encode the input tensors to latent vectors.
        Args:
            _: tf.Tensor, [B, ...], input tensors.
        Returns:
            _: tf.Tensor, [B, latent_dim], latent vectors.
        """
        return self.styleonly(*args, **kwargs)

    def mapper(self):
        """Model for mpping latent from prior.
        Returns:
            tf.keras.Model: map prior to latent.
        """
        lmap = LatentMap(num_layer=self.map_num_layer,
                         latent_dim=self.latent_dim,
                         hidden_dim=self.latent_dim)
        lmap.build((None, self.latent_dim))
        return lmap

    def generator(self):
        """Model for generating data from encoded latent.
        Returns:
            tf.keras.Model: generate sample from encoded latent.
        """
        gen = Generator(init_channels=self.init_channels,
                        max_channels=self.max_channels,
                        num_layer=self.num_layer,
                        out_channels=self.out_channels)
        gen.build((None, self.latent_dim))
        return gen

    def encoder(self):
        """Model for encoding data to fixed length latent vector.
        Returns:
            tf.keras.Model: encode data to fixed length latent vector.
        """
        enc = Encoder(init_channels=self.init_channels,
                      max_channels=self.max_channels,
                      num_layer=self.num_layer,
                      latent_dim=self.latent_dim)
        enc.build((None, self.img_size, self.img_size, self.out_channels))
        return enc

    def discriminator(self):
        """Model for discriminating real sample from fake one.
        Returns:
            tf.keras.Model: discriminate real sample from fake one.
        """
        disc = tf.keras.layers.Conv2D(1, 1, use_bias=False)
        channels = min(
            self.max_channels, self.init_channels * 2 ** self.num_layer)
        disc.build((None, 4, 4, channels))
        return disc

    @staticmethod
    def default_setting():
        """Default settings.
        Returns:
            Dict[str, Any], settings.
        """
        return {
            'latent_dim': 256,
            'num_layers': 4,
            'map_num_layers': 5,
            'init_channels': 32,
            'max_channels': 256,
            'out_channels': 1,
            'lr': 0.002,
            'beta1': 0.0,
            'beta2': 0.99,
            'gamma': 10,
        }

    class StyleOnly(tf.keras.Model):
        """Submodule for passing only style vector from encoder outputs.
        """
        def __init__(self, encoder):
            super(StyleAlae.StyleOnly, self).__init__()
            self.encoder = encoder
        
        def call(self, x):
            _, style = self.encoder(x)
            return style

    class Binder(tf.keras.Model):
        """Bind encoder outputs to discriminator.
        """
        def __init__(self, encoder, discriminator):
            super(StyleAlae.Binder, self).__init__()
            self.encoder = encoder
            self.discriminator = discriminator

        def call(self, x):
            x, _ = self.encoder(x)
            return self.discriminator(x)
