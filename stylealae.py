import tensorflow as tf


class AffineTransform(tf.keras.Model):
    def __init__(self, shape):
        super(AffineTransform, self).__init__()
        self.shape = shape
        self.weight = tf.Variable(tf.random.normal(self.shape))
        self.bias = tf.Variable(tf.zeros(self.shape))

    def call(self, x):
        """
        Args:
            x: tf.Tensor, arbitary shape
        Returns:
            tf.Tensor, arbitary shape
        """
        return x * self.weight + self.bias


class Normalize2D(tf.keras.Model):
    def __init__(self, eps=1e-8):
        super(Normalize2D, self).__init__()
        self.eps = eps

    def call(self, x):
        """
        Args:
            x: tf.Tensor, [B, H, W, C]
        Returns:
            tf.Tensor, [B, H, W, C]
        """
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        return (x - mean) / (tf.math.sqrt(var) + self.eps)


class Repeat2D(tf.keras.Model):
    def __init__(self, factor):
        super(Repeat2D, self).__init__()
        self.factor = factor
    
    def call(self, x):
        """
        Args:
            x: tf.Tensor, [B, H, W, C]
        Returns:
            tf.Tensor, [B, H x factor, W x factor, C]
        """
        shape = tf.shape(x)
        x = x[:, :, None, :, None, :]
        x = tf.tile(x, [1, 1, self.factor, 1, self.factor, 1])
        return tf.reshape(x, [shape[0], shape[1] * self.factor, shape[2] * self.factor, shape[3]])


class LatentMap(tf.keras.Model):
    def __init__(self, num_layer, z_dim, latent_dim, hidden_dim):
        super(LatentMap, self).__init__()
        self.num_layer = num_layer
        self.z_dim = z_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.leaky_relu = tf.keras.layers.LeakyReLU(0.2)
        self.blocks = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation=self.leaky_relu)
            for _ in range(self.num_layer - 1)])

        self.blocks.add(
            tf.keras.layers.Dense(2 * self.latent_dim))

        self.blocks.build([None, self.z_dim])

    def call(self, z):
        """
        Args:
            z: tf.Tensor, [B, z_dim]
        Returns:
            tf.Tensor, [B, 2, latent_dim]
        """
        latent = self.blocks(z)
        return tf.reshape(latent, [-1, 2, self.latent_dim])


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
            channels //= 2
            resolution *= 2
            out_dim = min(self.max_channels, channels)
            self.blocks.append(
                Generator.Block(out_dim,
                                i > 0,
                                'repeat' if resolution < 128 else 'deconv'))

        self.postconv = tf.keras.layers.Conv2D(self.out_channels, 1)
    
    def call(self, styles):
        """
        Args:
            styles: tf.Tensor, [B, num_layer * 2, latent_dim]
        Returns:
            tf.Tensor, [B, S, S, out_channels],
                where S = 4 * 2 ** num_layer
        """
        # [B, 4, 4, in_dim]
        x = self.const
        for i in range(self.num_layer):
            # [B, S', S', C] where S' = 4 * 2 ** (i + 1), C = in_dim / (2 ** (i + 1))
            x = self.blocks[i](x, styles[:, i * 2], styles[:, i * 2 + 1])
        # [B, S, S, out_channels]
        return self.postconv(x)

    class Block(tf.keras.Model):
        def __init__(self, out_dim, preconv, upsample):
            self.out_dim = out_dim
            self.preconv = preconv
            self.upsample = upsample

            if self.preconv:
                if self.upsample == 'repeat':
                    self.upsample_conv = \
                        tf.keras.Sequential([
                            Repeat2D(2),
                            tf.keras.layers.Conv2D(
                                self.out_dim,
                                kernel_size=3,
                                strides=1,
                                padding='same',
                                use_bias=False)])
                elif self.upsample == 'deconv':
                    self.upsample_conv = \
                        tf.keras.layers.Conv2DTranspose(
                            self.out_dim,
                            kernel_size=3,
                            strides=2,
                            padding='same',
                            use_bias=False)
    
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


class Encoder(tf.keras.Model):
    def __init__(self,
                 init_channels,
                 max_channels,
                 num_layer,
                 latent_dim,
                 out_channels):
        super(Encoder, self).__init__()
        self.init_channels = init_channels
        self.max_channels = max_channels
        self.num_layer = num_layer
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        resolution = 4 * 2 ** self.num_layer
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
                              i == self.num_layer - 1,
                              'pool' if resolution < 128 else 'conv'))

        self.disc = tf.keras.layers.Dense(1)

    class Block(tf.keras.Model):
        def __init__(self, out_dim, last, downsample):
            super(Encoder.Block, self).__init__()
            self.out_dim = out_dim
            self.last = last
            self.downsample = downsample

            self.preconv = tf.keras.layers.Conv2D(
                
            )

            if self.last:
                self.dense = tf.keras.layers.Dense(out_dim)
            else:
                if self.downsample == 'pool':
                    self.downsample_conv = tf.keras.Sequential([
                        tf.keras.layers.Conv2D(
                            out_dim, 3, 1, padding='same', use_bias=False),
                        tf.keras.layers.AveragePooling2D(2)])
                else:
                    self.downsample_conv = tf.keras.layers.Conv2D(
                        out_dim, 3, 2, padding='same', use_bias=False)


