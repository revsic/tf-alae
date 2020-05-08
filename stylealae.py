import tensorflow as tf


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
            tf.keras.layers.Dense(self.latent_dim))

        self.blocks.build((None, self.z_dim))

    def call(self, z):
        return self.blocks(z)


class Generator(tf.keras.Model):
    def __init__(self,
                 init_channels,
                 max_channels,
                 num_layer,
                 latent_dim,
                 out_channels):
        super(Generator, self).__init__()
        self.init_channels = init_channels
        self.max_channels = max_channels
        self.num_layer = num_layer
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        resolution = 4
        channels = self.init_channels * 2 ** (self.num_layer - 1)
        in_dim = min(self.max_channels, channels)
        self.const = tf.Variable(
            tf.ones((1, in_dim, resolution, resolution)),
            dtype=tf.float32)

        self.blocks = []
        for i in range(self.num_layer):
            out_dim = min(self.max_channels, channels)
            self.blocks.append(
                Generator.Block(in_dim,
                                out_dim,
                                i > 0,
                                'repeat' if resolution < 128 else 'deconv'))

            in_dim = out_dim
            channels //= 2
            resolution *= 2

        self.rgb = tf.keras.layers.Conv2D(self.out_channels, 1)
    
    def call(self, styles, noise):
        x = self.const
        for i in range(self.num_layer):
            x = self.blocks[i](x, styles[:, i * 2], styles[:, i * 2 + 1], noise)

        return self.rgb(x)

    class Block(tf.keras.Model):
        def __init__(self, in_dim, out_dim, preconv, upsample):
            super(Generator.Block, self).__init__()
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.preconv = preconv
            self.upsample = upsample

            if self.preconv:
                if self.upsample == 'repeat':
                    pass
                elif self.upsample == 'deconv':
                    pass
            
            

class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
