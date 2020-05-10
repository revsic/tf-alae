import tensorflow as tf

from .utils import AffineTransform, Normalize2D, Repeat2D


class Generator(tf.keras.Model):
    def __init__(self,
                 init_channels,
                 max_channels,
                 num_layer,
                 out_channels):
        super(Generator, self).__init__()
        self.init_channels = init_channels
        self.max_channels = max_channels
        self.num_layer = num_layer
        self.out_channels = out_channels

        resolution = 4
        channels = self.init_channels * 2 ** (self.num_layer - 1)
        out_dim = min(self.max_channels, channels)
        self.const = tf.Variable(
            tf.ones([1, resolution, resolution, out_dim]),
            dtype=tf.float32)

        self.blocks = []
        for i in range(self.num_layer):
            out_dim = min(self.max_channels, channels)
            self.blocks.append(
                Generator.Block(out_dim,
                                i > 0,
                                'repeat' if resolution < 128 else 'deconv'))
            channels //= 2
            resolution *= 2

        self.postconv = tf.keras.layers.Conv2D(self.out_channels, 1)
    
    def call(self, styles):
        """
        Args:
            styles: tf.Tensor, [B, latent_dim]
        Returns:
            tf.Tensor, [B, S, S, out_channels],
                where S = 4 * 2 ** (num_layer - 1)
        """
        # [B, 4, 4, in_dim]
        x = self.const
        for block in self.blocks:
            # [B, S', S', C] where S' = 4 * 2 ** i, C = in_dim / (2 ** (i + 1))
            x = block(x, styles, styles)
        # [B, S, S, out_channels]
        return self.postconv(x)

    class Block(tf.keras.Model):
        def __init__(self, out_dim, preconv, upsample):
            super(Generator.Block, self).__init__()
            self.out_dim = out_dim
            self.preconv = preconv
            self.upsample = upsample

            if self.preconv:
                if self.upsample == 'repeat':
                    self.upsample_conv = tf.keras.Sequential([
                        Repeat2D(2),
                        tf.keras.layers.Conv2D(
                            self.out_dim, 3,
                            strides=1,
                            padding='same',
                            use_bias=False)])
                elif self.upsample == 'deconv':
                    self.upsample_conv = tf.keras.layers.Conv2DTranspose(
                        self.out_dim, 3, 2, padding='same', use_bias=False)
    
            self.leaky_relu = tf.keras.layers.LeakyReLU(0.2)
            self.normalize = Normalize2D()

            self.noise_affine1 = AffineTransform([1, 1, 1, self.out_dim])
            self.latent_proj1 = tf.keras.layers.Dense(self.out_dim * 2)

            self.conv = tf.keras.layers.Conv2D(
                self.out_dim, 3, 1, padding='SAME', use_bias=False)
            
            self.noise_affine2 = AffineTransform([1, 1, 1, self.out_dim])
            self.latent_proj2 = tf.keras.layers.Dense(self.out_dim * 2)

        def call(self, x, s1, s2):
            """
            Args:
                x: tf.Tensor, [B, H, W, in_dim]
                s1: tf.Tensor, [B, latent_dim]
                s2: tf.Tensor, [B, latent_dim]
            Returns:
                tf.Tensor, [B, Hx2, Wx2, out_dim]
            """
            if self.preconv:
                # [B, Hx2, Wx2, out_dim]
                x = self.upsample_conv(x)

            shape = tf.shape(x)
            # [1, Hx2, Wx2, 1]
            noise = tf.random.normal([1, shape[1], shape[2], 1])
            # [B, Hx2, Wx2, out_dim]
            x = self.leaky_relu(x + self.noise_affine1(noise))
            # [B, Hx2, Wx2, out_dim]
            x = self.normalize(x)

            # [B, out_dim x 2]
            s1 = self.latent_proj1(s1)
            # [B, 2, out_dim]
            s1 = tf.reshape(s1, [-1, 2, self.out_dim])
            # [B, Hx2, Wx2, out_dim]
            x = s1[:, 0, None, None, :] + x * s1[:, 1, None, None, :]

            # [B, Hx2, Wx2, out_dim]
            x = self.conv(x)

            # [1, Hx2, Wx2, 1]
            noise = tf.random.normal([1, shape[1], shape[2], 1])
            # [B, Hx2, Wx2, out_dim]
            x = self.leaky_relu(x + self.noise_affine2(noise))
            # [B, Hx2, Wx2, out_dim]
            x = self.normalize(x)

            # [B, out_dim x 2]
            s2 = self.latent_proj2(s2)
            # [B, 2, out_dim]
            s2 = tf.reshape(s2, [-1, 2, self.out_dim])
            # [B, Hx2, Wx2, out_dim]
            x = s2[:, 0, None, None, :] + x * s1[:, 1, None, None, :]

            return x
