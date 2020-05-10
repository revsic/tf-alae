import tensorflow as tf

from ..alae import ALAE
from .encoder import Encoder
from .generator import Generator
from .maplatent import LatentMap


class StyleAlae(ALAE):
    def __init__(self, settings):
        super(ALAE, self).__init__()
        self.settings = settings
        self.latent_dim = self.settings['latent_dim']
        self.num_layers = self.settings['num_layers']
        self.map_num_layers = self.settings['map_num_layers']
        self.init_channels = self.settings['init_channels']
        self.max_channels = self.settings['max_channels']
        self.out_channels = self.settings['out_channels']

        self.latent_map = LatentMap(num_layer=self.map_num_layers,
                                    z_dim=self.latent_dim,
                                    latent_dim=self.latent_dim,
                                    hidden_dim=self.latent_dim)
        self.generator = Generator(init_channels=self.init_channels,
                                   max_channels=self.max_channels,
                                   num_layer=self.num_layers,
                                   out_channels=self.out_channels)
        self.encoder = Encoder(init_channels=self.init_channels,
                               max_channels=self.max_channels,
                               num_layer=self.num_layers,
                               latent_dim=self.latent_dim)
        self.discriminator = tf.keras.layers.Dense(1)

        self.fakepass = tf.keras.Sequential([
            self.latent_map, self.generator, self.encoder, self.discriminator])
        self.realpass = tf.keras.Sequential([self.encoder, self.discriminator])
        self.latentpass = tf.keras.Sequential([
            self.latent_map, self.generator, self.encoder])

        

    def encoder(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def generator(self, *args, **kwargs):
        return self.generator(*args, **kwargs)

    def losses(self, x):
        pass

    def trainstep(self, x):
        pass

    @staticmethod
    def default_setting():
        return {
            'latent_dim': 256,
            'num_layers': 3,
            'map_num_layers': 5,
            'init_channels': 32,
            'max_channels': 256,
            'out_channels': 3,
        }
