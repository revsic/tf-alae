import tensorflow as tf

from .utils import AffineTransform, Normalize2D, Repeat2D


class Encoder(tf.keras.Model):
    def __init__(self,
                 init_channels,
                 max_channels,
                 num_layer,
                 latent_dim):
        super(Encoder, self).__init__()
        self.init_channels = init_channels
        self.max_channels = max_channels
        self.num_layer = num_layer
        self.latent_dim = latent_dim

        resolution = 4 * 2 ** (self.num_layer - 1)
        channels = self.init_channels
        out_dim = min(self.max_channels, channels)
        self.preconv = tf.keras.layers.Conv2D(out_dim, 1)

        self.blocks = []
        for i in range(self.num_layer):
            channels *= 2
            resolution //= 2
            out_dim = min(self.max_channels, channels)
            self.blocks.append(
                Encoder.Block(out_dim,
                              self.latent_dim,
                              i > 0,
                              'pool' if resolution < 128 else 'conv'))

    def call(self, x):
        bsize = x.shape[0]
        styles = tf.zeros([bsize, self.latent_dim])

        x = self.preconv(x)
        for block in self.blocks:
            x, s1, s2 = block(x)
            styles += s1 + s2

        return x, styles

    class Block(tf.keras.Model):
        def __init__(self, out_dim, latent_dim, preconv, downsample):
            super(Encoder.Block, self).__init__()
            self.out_dim = out_dim
            self.latent_dim = latent_dim
            self.preconv = preconv
            self.downsample = downsample

            if self.preconv:
                if self.downsample == 'pool':
                    self.downsample_conv = tf.keras.Sequential([
                        tf.keras.layers.Conv2D(
                            out_dim, 3, 1, padding='same', use_bias=False),
                        tf.keras.layers.AveragePooling2D(2)])
                else:
                    self.downsample_conv = tf.keras.layers.Conv2D(
                        out_dim, 3, 2, padding='same', use_bias=False)

            self.leaky_relu = tf.keras.layers.LeakyReLU(0.2)
            self.normalize = Normalize2D()

            self.conv = tf.keras.layers.Conv2D(
                self.out_dim, 3, 1, padding='same', use_bias=False)

            self.style_proj1 = tf.keras.layers.Dense(self.latent_dim)
            self.style_proj2 = tf.keras.layers.Dense(self.latent_dim)

        def call(self, x):
            """
            Args:
                x: tf.Tensor, [B, H, W, in_dim]
            Returns:
                x: tf.Tensor, [B, H/2, W/2, out_dim]
                style1: tf.Tensor, [B, latent_dim]
                style2: tf.Tensor, [B, latent_dim]
            """
            if self.preconv:
                # [B, H/2, W/2, out_dim]
                x = self.downsample_conv(x)
                x = self.leaky_relu(x)

            # [B, out_dim]
            mean, var = tf.nn.moments(x, axes=[1, 2])
            # [B, out_dim * 2]
            stat = tf.concat([mean, var], axis=-1)
            # [B, latent_dim]
            style1 = self.style_proj1(stat)

            # [B, H/2, W/2, out_dim]
            x = self.normalize(x)
            # [B, H/2, W/2, out_dim]
            x = self.conv(x)
            x = self.leaky_relu(x)

            # [B, out_dim]
            mean, var = tf.nn.moments(x, axes=[1, 2])
            # [B, out_dim * 2]
            stat = tf.concat([mean, var], axis=-1)
            # [B, latent_dim]
            style2 = self.style_proj2(stat)

            # [B, H/2, W/2, out_dim]
            x = self.normalize(x)

            return x, style1, style2
