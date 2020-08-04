import tensorflow as tf

from .utils import AffineTransform, Repeat2D, Blur, normalize2d


class Generator(tf.keras.Model):
    """Generate image from style vector.
    """
    def __init__(self,
                 init_channels,
                 max_channels,
                 num_layer,
                 out_channels):
        """Initializer.
        Args:
            init_channels: int, number of the channels of global latent.
            max_channels: int, maximum channels of the feature map.
            num_layer: int, number of the layers, it determines the size of the image.
            out_channels: int, number of the image channels, 3 for rgb, 1 for gray-scale.
        """
        super(Generator, self).__init__()
        self.init_channels = init_channels
        self.max_channels = max_channels
        self.num_layer = num_layer
        self.out_channels = out_channels
        self.level = self.num_layer - 1

        resolution = 4
        channels = self.init_channels * 2 ** (self.num_layer - 1)
        out_dim = min(self.max_channels, channels)
        self.const = tf.ones([1, resolution, resolution, out_dim])

        self.blocks = []
        self.to_rgb = []
        for i in range(self.num_layer):
            self.blocks.append(
                Generator.Block(out_dim,
                                i > 0,
                                'repeat' if resolution < 128 else 'deconv'))
            # intermediates to rgb
            self.to_rgb.append(tf.keras.layers.Conv2D(self.out_channels, 1))
            # update layer infos for next
            channels //= 2
            resolution *= 2
            out_dim = min(self.max_channels, channels)
    
    def set_level(self, level):
        """Set training level, start from first block to last block.
        Args:
            level: int, training level, in range [0, num_layer).
        """
        self.level = level
    
    def level_variables(self):
        """Get trainable variables based on training level.
        Returns:
            List[tf.Variable], trainable variables.
        """
        var = self.to_rgb[self.level].trainable_variables
        for block in self.blocks[:self.level + 1]:
            var += block.trainable_variables
        return var

    def call(self, styles):
        """Generate image.
        Args:
            styles: tf.Tensor, [B, latent_dim], style vector.
        Returns:
            tf.Tensor, [B, S, S, out_channels], generated image.
                where S = 2 ** (num_layer + 1).
                if self.level is set, S = 2 ** (level + 2).
        """
        # [B, 4, 4, in_dim]
        x = self.const
        for block in self.blocks[:self.level + 1]:
            # [B, S', S', C] where S' = 4 * 2 ** i, C = in_dim / (2 ** (i + 1))
            x = block(x, styles, styles)
        # [B, S, S, out_channels]
        return self.to_rgb[self.level](x)

    class Block(tf.keras.Model):
        """Generator block for progressive growing.
        """
        def __init__(self, out_dim, upsample, policy=None):
            """Initializer.
            Args:
                out_dim: int, number of channels of output feature map.
                upsample: bool, whether run convolution for upsampling.
                policy: Optional[str], upsampling policy for pre-convolution.
                    - repeat: (2x2, repeat)-upsample -> (3x3, stride=1)-conv
                    - deconv: (3x3, stride=2)-conv2d transpose
            """
            super(Generator.Block, self).__init__()
            self.out_dim = out_dim
            self.upsample = upsample
            self.policy = policy

            if not self.upsample:
                self.conv1 = tf.keras.layers.Conv2D(
                    self.out_dim, 3, strides=1, padding='same', use_bias=False)
            elif self.policy == 'repeat':
                self.conv1 = tf.keras.Sequential([
                    Repeat2D(2),
                    tf.keras.layers.Conv2D(
                        self.out_dim, 3, 1, padding='same', use_bias=False)])
            elif self.policy == 'deconv':
                self.conv1 = tf.keras.layers.Conv2DTranspose(
                    self.out_dim, 3, strides=2, padding='same', use_bias=False)
            else:
                raise ValueError('invalid upsample, policy arguments pair')
    
            self.conv2 = tf.keras.layers.Conv2D(
                self.out_dim, 3, 1, padding='SAME', use_bias=False)
            
            self.blur = Blur()

            self.noise_affine1 = AffineTransform([1, 1, 1, self.out_dim])
            self.latent_proj1 = tf.keras.layers.Dense(self.out_dim * 2)

            self.noise_affine2 = AffineTransform([1, 1, 1, self.out_dim])
            self.latent_proj2 = tf.keras.layers.Dense(self.out_dim * 2)

        def call(self, x, s1, s2):
            """Generate next level feature map.
            Args:
                x: tf.Tensor, [B, H, W, in_dim], input feature map.
                s1: tf.Tensor, [B, latent_dim], first style.
                s2: tf.Tensor, [B, latent_dim], second style.
                noise: tf.Tensor, [1, H', W', 1], spatial noise tensor.
            Returns:
                tf.Tensor, [B, H', W', out_dim], colored feature map,
                    H' = Hx2 if upsample else H,
                    W' = Wx2 if upsample else W.
            """
            # [B, H', W', out_dim]
            x = self.conv1(x)
            x = self.blur(x)
            # [B, H', W', out_dim]
            x = self.apply_noise(x, self.noise_affine1)
            # [B, H', W', out_dim]
            x = self.apply_style(x, style1, self.latent_proj1)

            # [B, H', W', out_dim]
            x = self.conv2(x)
            x = self.blur(x)
            # [B, H', W', out_dim]
            x = self.add_noise(x, self.noise_affine2)
            # [B, H', W', out_dim]
            x = self.apply_style(x, style2, self.latent_proj2)
            return x

        def apply_noise(self, x, affine):
            """Apply spatial noise to feature map.
            Args:
                x: tf.Tensor, [B, H, W, C], input tensor.
                affine: tf.keras.Model, affine transformation layer for noise vector,
                    Callable, tf.Tensor[..., 1] -> tf.Tensor[..., C]
            Returns:
                x: tf.Tensor, [B, H, W, C], applied tensor.
            """
            # B, H, W, C
            shape = tf.shape(x)
            # [1, H, W, 1]
            noise = tf.random.normal([1, shape[1], shape[2], 1])
            # [B, H, W, C]
            x = tf.nn.leaky_relu(x + affine(noise), alpha=0.2)
            return x

        def apply_style(self, x, style, proj, eps=1e-8):
            """Apply channel-wise styles to feature map.
            Args:
                x: tf.Tensor, [B, H, W, C], input tensor.
                style: tf.Tensor, [B, latent_dim], style vector.
                proj: tf.keras.Model, style projection layer,
                    Callable, tf.Tensor[..., latent_dim] -> tf.Tensor[..., Cx2]
            Returns:
                x: tf.Tensor, [B, H, W, C], colored tensor.
            """
            channels = x.shape[-1]
            # [B, Cx2]
            style = proj(style)
            # [B, C]
            mu, log_sigma = style[:, :channels], style[:, channels:]
            # [B, H, W, C]
            x = normalize2d(x, eps=eps)
            # [B, H, W, C]
            return mu[:, None, None] + tf.exp(log_sigma[:, None, None]) * x
