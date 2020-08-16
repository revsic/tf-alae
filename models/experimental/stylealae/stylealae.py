import numpy as np
import tensorflow as tf

from ...alae import ALAE
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
        self.disc_num_layer = self.settings['disc_num_layers']
        self.init_channels = self.settings['init_channels']
        self.max_channels = self.settings['max_channels']
        self.out_channels = self.settings['out_channels']

        self.level = 0
        self.img_size = 2 ** (self.num_layer + 1)

        self.prepare(self.latent_dim,
                     self.settings['gamma'],
                     self.settings['lr'],
                     self.settings['beta2'],
                     self.settings['beta2'],
                     self.settings['disc_gclip'])
        
        self.rctor_opt = tf.keras.optimizers.Adam(
            self.settings['lr'], self.settings['beta1'], self.settings['beta2'])

    def set_level(self, level):
        """Set training level for progressive growing.
        Args:
            level: int, training level, in range [0, num_layer).
        """
        self.level = level
        self.gen.set_level(self.level)
        self.enc.set_level(self.level)

        gen_var = self.gen.level_variables()
        enc_var = self.enc.level_variables()
        self.ed_var = enc_var + self.disc.trainable_variables
        self.fg_var = self.map.trainable_variables + gen_var
        self.eg_var = enc_var + gen_var

    def preproc(self, x, level=None):
        """Preprocess inputs to resize with respects to level.
        Args:
            x: tf.Tensor, [B, H, W, C], image tensor.
            level: int, resolution level.
        Returns:
            tf.Tensor, [B, S, S, C], downsampled image
                where S = 2 ** (level + 2).
        """
        if level is None:
            level = self.level
        size = 2 ** (level + 2)
        x = tf.image.resize(x, [size, size], antialias=True)
        return x

    def encode(self, x):
        """Encode the input tensors to latent vectors.
        Args:
            x: tf.Tensor, [B, H, W, C], image tensor.
        Returns:
            tf.Tensor, [B, latent_dim], latent_vectors
        """
        return self.enc(self.preproc(x))

    # @tf.function
    # def _rctor_loss(self, _, x):
    #     rctor = self.gen(self.enc(x))
    #     return tf.reduce_mean(tf.abs(rctor - x))

    def losses(self, x):
        x = self.preproc(x)
        return super(StyleAlae, self).losses(x)

    def trainstep(self, x):
        x = self.preproc(x)
        return super(StyleAlae, self).trainstep(x)

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
        for i in range(self.num_layer, 0, -1):
            gen.set_level(i - 1)
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
        
        img_size = self.img_size
        for i in range(self.num_layer, 0, -1):
            enc.set_level(i - 1)
            enc.build((None, img_size, img_size, self.out_channels))
            img_size //= 2
        return enc

    def discriminator(self):
        """Model for discriminating real sample from fake one.
        Returns:
            tf.keras.Model: discriminate real sample from fake one.
        """
        disc = LatentMap(num_layer=self.disc_num_layer,
                         latent_dim=1,
                         hidden_dim=self.latent_dim)
        disc.build((None, self.latent_dim))
        return disc

    @staticmethod
    def default_setting(imgsize=256):
        """Default settings.
        Returns:
            Dict[str, Any], settings.
        """
        num_layers = int(np.log2(imgsize)) - 1
        return {
            'latent_dim': 512,
            'num_layers': num_layers,
            'map_num_layers': 8,
            'disc_num_layers': 3,
            'init_channels': 32,
            'max_channels': 512,
            'out_channels': 3,
            'lr': 0.001,
            'beta1': 0.9,
            'beta2': 0.99,
            'gamma': 10,
            'disc_gclip': None,
        }
