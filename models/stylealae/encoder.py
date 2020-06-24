import tensorflow as tf

from .utils import AffineTransform, Normalize2D, Repeat2D


class Encoder(tf.keras.Model):
    """Encode style vector from image.
    """
    def __init__(self,
                 init_channels,
                 max_channels,
                 num_layer,
                 latent_dim):
        """Initializer.
        Args:
            init_channels: int, the number of channels of the global latent.
            max_channels: int, maximum channels of the features.
            num_layer: int, the number of the layers, it determines the size of global latent.
            latent_dim: int, size of the style vector.
        """
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
        """Generate style vector and global latent x.
        Args:
            x: tf.Tensor, [B, H, W, C], image tensor.
        Returns:
            styles: tf.Tensor, [B, latent_dim], style vector.
        """
        bsize = tf.shape(x)[0]
        styles = tf.zeros([bsize, self.latent_dim], dtype=tf.float32)

        x = self.preconv(x)
        for block in self.blocks:
            x, s1, s2 = block(x)
            styles += s1 + s2

        return styles

    class Block(tf.keras.Model):
        """Encoder block for progressive downsampling.
        """
        def __init__(self, out_dim, latent_dim, preconv, downsample):
            """Initializer.
            Args:
                out_dim: int, number of the output channels.
                latent_dim: int, size of the latent vector.
                preconv: bool, whether run convolution for downsampling or not.
                downsample: str, down sampling policy for pre-convolution options.
                    - pool: (3x3, stride=1)-conv -> (2x2)-avgpool
                    - conv: (3x3, stride=2)-conv
            """
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
            """Generate style vector.
            Args:
                x: tf.Tensor, [B, H, W, in_dim], input feature map.
            Returns:
                x: tf.Tensor, [B, H/2, W/2, out_dim], whitened feature map.
                style1: tf.Tensor, [B, latent_dim], first style.
                style2: tf.Tensor, [B, latent_dim], second style.
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
