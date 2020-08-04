import tensorflow as tf

from .utils import AffineTransform, Normalize2D, Repeat2D, Blur


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
        self.level = self.num_layer - 1

        resolution = 4 * 2 ** (self.num_layer - 1)
        channels = self.init_channels
        out_dim = min(self.max_channels, channels)

        self.blocks = []
        self.from_rgb = []
        for i in range(self.num_layer):
            # rgb to intermediates
            self.from_rgb.append(tf.keras.layers.Conv2D(out_dim, 1))
            # update layer infos
            channels *= 2
            resolution //= 2
            in_dim = out_dim
            out_dim = min(self.max_channels, channels)
            # add block
            self.blocks.append(
                Encoder.Block(in_dim, out_dim,
                              self.latent_dim,
                              i < self.num_layer - 1,
                              'pool' if resolution < 128 else 'conv'))

    def set_level(self, level):
        """Set training level, start from last block to first block.
        Args:
            level: int, training level, in range [0, num_layer).
        """
        self.level = level

    def level_variables(self):
        """Get trainable variables based on training level.
        Returns:
            List[tf.Variable], trainable variables.
        """
        start = self.num_layer - 1 - self.level
        var = self.from_rgb[start].trainable_variables
        for block in self.blocks[start:]:
            var += block.trainable_variables
        return var

    def call(self, x):
        """Generate style vector and global latent x.
        Args:
            x: tf.Tensor, [B, H, W, C], image tensor.
        Returns:
            styles: tf.Tensor, [B, latent_dim], style vector.
        """
        bsize = tf.shape(x)[0]
        styles = tf.zeros([bsize, self.latent_dim], dtype=tf.float32)

        start = self.num_layer - self.level - 1

        x = self.from_rgb[start](x)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        for block in self.blocks[start:]:
            x, s1, s2 = block(x)
            styles += s1 + s2

        return styles

    class Block(tf.keras.Model):
        """Encoder block for progressive downsampling.
        """
        def __init__(self, in_dim, out_dim, latent_dim, downsample, policy=None):
            """Initializer.
            Args:
                in_dim: int, number of the input channels.
                out_dim: int, number of the output channels.
                latent_dim: int, size of the latent vector.
                downsample: bool, whether run convolution for downsampling or not.
                policy: Optional[str], down sampling policyfor pre-convolution options.
                    - pool: (3x3, stride=1)-conv -> (2x2)-avgpool
                    - conv: (3x3, stride=2)-conv
            """
            super(Encoder.Block, self).__init__()
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.latent_dim = latent_dim
            self.downsample = downsample
            self.policy = policy

            self.conv1 = tf.keras.layers.Conv2D(
                self.in_dim, 3, 1, padding='SAME', use_bias=False)

            if self.downsample:
                if self.policy == 'pool':
                    conv2 = tf.keras.Sequential([
                        tf.keras.layers.Conv2D(
                            self.out_dim, 3, 1, padding='SAME', use_bias=False),
                        tf.keras.layers.AveragePooling2D(2)])
                elif self.policy == 'conv':
                    conv2 = tf.keras.layers.Conv2D(
                        self.out_dim, 3, 2, padding='SAME', use_bias=False)
                else:
                    raise ValueError('invalid argument `policy`')

                self.conv2 = tf.keras.Sequential([conv2, Blur()])

            self.normalize = Normalize2D()

            self.style_proj1 = tf.keras.layers.Dense(self.latent_dim)
            self.style_proj2 = tf.keras.layers.Dense(self.latent_dim)

        def call(self, x):
            """Generate style vector.
            Args:
                x: tf.Tensor, [B, H, W, in_dim], input feature map.
            Returns:
                x: tf.Tensor, [B, H', W', C], whitened feature map,
                    H' = H/2 if downsample else H,
                    W' = W/2 if downsample else W.
                    C  = out_dim if downsample else in_dim.
                style1: tf.Tensor, [B, latent_dim], first style.
                style2: tf.Tensor, [B, latent_dim], second style.
            """
            # [B, latent_dim]
            style1 = self.extract_style(x, self.style_proj1)
            # [B, H, W, in_dim]
            x = self.normalize(x)
            # [B, H, W, in_dim]
            x = tf.nn.leaky_relu(self.conv1(x), alpha=0.2)

            # [B, latent_dim]
            style2 = self.extract_style(x, self.style_proj2)
            if self.downsample:
                # [B, H, W, out_dim]
                x = self.normalize(x)
                # [B, H', W', out_dim]
                x = tf.nn.leaky_relu(self.conv2(x), alpha=0.2)

            return x, style1, style2

        def extract_style(self, x, proj, eps=1e-8):
            """Extract style vector from feature map.
            Args:
                x: tf.Tensor, [B, H, W, C], input tensor.
                proj: tf.keras.Model, projection layer,
                    Callable, tf.Tensor[..., Cx2] -> tf.Tensor[..., latent_dim]
            Returns:
                style: tf.Tensor, [B, latent_dim], style vector.
            """
            # [B, C]
            mean, var = tf.nn.moments(x, axes=[1, 2])
            # [B, C]
            log_sigma = tf.log(tf.sqrt(var) + eps)
            # [B, Cx2]
            stat = tf.concat([mean, log_sigma], axis=-1)
            # [B, latent_dim]
            return proj(stat)
