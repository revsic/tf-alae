import tensorflow as tf

from alae import ALAE


class MlpAlae(ALAE):
    """ALAE with multi-layer perceptron architecture.
    Attributes:
        z_dim: int, dimension of the latent prior.
        latent_dim: int, dimension of the latent vector.
        output_dim: int, dimension of the output vector.
        gamma: int, coefficient for gradient regularization term.
        f, g, e, d: tf.keras.Model, ALAE component,
            latent-map, generator, encoder, discriminator.
        fakepass, realpass, latentpass: tf.keras.Model,
            additional component for computing objectives.
        ed_var, fg_var, eg_var: List[tf.Tensor], trainable variables.
        ed_opt, fg_opt, eg_opt: tf.keras.optimizers.Optimizer,
            optimizers for training step.
    """
    def __init__(self, settings=None):
        """Constructor.
        Args:
            settings: Dict[str, any], model settings,
                reference `MlpAlae.default_setting`.
        """
        super(MlpAlae, self).__init__()
        if settings is None:
            settings = self.default_setting()

        self.z_dim = settings['z_dim']
        self.latent_dim = settings['latent_dim']
        self.output_dim = settings['output_dim']

        self.gamma = settings['gamma']

        def seq(dims, input_dim, activation=tf.nn.relu):
            layer = tf.keras.Sequential([
                tf.keras.layers.Dense(dim, activation=activation)
                for dim in dims[:-1]])
            layer.add(
                tf.keras.layers.Dense(dims[-1]))
            layer.build((None, input_dim))
            return layer

        self.f = seq(settings['f'], self.z_dim)
        self.g = seq(settings['g'], self.latent_dim)
        self.e = seq(settings['e'], self.output_dim)
        self.d = seq(settings['d'], self.latent_dim)

        self.fakepass = tf.keras.Sequential([
            self.f, self.g, self.e, self.d])
        self.realpass = tf.keras.Sequential([self.e, self.d])
        self.latentpass = tf.keras.Sequential([self.f, self.g, self.e])

        self.ed_var = self.e.trainable_variables + self.d.trainable_variables
        self.fg_var = self.f.trainable_variables + self.g.trainable_variables
        self.eg_var = self.e.trainable_variables + self.g.trainable_variables

        self.ed_opt = tf.keras.optimizers.Adam(
            settings['lr'], settings['beta1'], settings['beta2'])
        self.fg_opt = tf.keras.optimizers.Adam(
            settings['lr'], settings['beta1'], settings['beta2'])
        self.eg_opt = tf.keras.optimizers.Adam(
            settings['lr'], settings['beta1'], settings['beta2'])

    def encoder(self, *args, **kwargs):
        """Encode the input tensors to latent vectors.
        Args:
            _: tf.Tensor, [B, output_dim], input tensors.
        Returns:
            _: tf.Tensor, [B, latent_dim], latent vectors.
        """
        return self.e(*args, **kwargs)

    def generator(self, *args, **kwargs):
        """Generate output tensors from latent vectors.
        Args:
            _: tf.Tensor, [B, latent_dim], latent vectors.
        Returns:
            _: tf.Tensor, [B, output_dim], output tensors.
        """
        return self.g(*args, **kwargs)

    @tf.function
    def _disc_loss(self, z, x):
        """Compute discriminator loss.
        Args:
            z: tf.Tensor, [B, z_dim], latent prior.
            x: tf.Tensor, [B, output_dim], output tensors.
        Returns:
            tf.Tensor, [], loss value.
        """
        with tf.GradientTape() as tape:
            fakeloss = tf.reduce_mean(tf.math.softplus(self.fakepass(z)))
            realloss = tf.reduce_mean(tf.math.softplus(-self.realpass(x)))

        # gradient regularizer
        grad = tape.gradient(realloss, self.ed_var)
        gradreg = self.gamma / 2 * tf.reduce_mean([
            tf.reduce_mean(tf.square(g)) for g in grad])

        return fakeloss + realloss + gradreg

    @tf.function
    def _gen_loss(self, z, _=None):
        """Compute generator loss.
        Args:
            z: tf.Tensor, [B, z_dim], latent prior.
            _: unused, placeholder.
        Returns:
            tf.Tensor, [], generator loss value.
        """
        return tf.reduce_mean(tf.math.softplus(-self.fakepass(z)))

    @tf.function
    def _latent_loss(self, z, _=None):
        """Compute latent loss.
        Args:
            z: tf.Tensor, [B, z_dim], latent prior.
            _: unused, placeholder.
        Returns:
            tf.Tensor, [], latent loss value.
        """
        latent = self.f(z)
        recovered = self.latentpass(z)
        return tf.reduce_mean(tf.square(latent - recovered))

    def losses(self, x):
        """Loss values for tensorboard summary.
        Args:
            x: tf.Tensor, [B, output_dim], output samples.
        Returns:
            Dict[str, np.array], loss values.
        """
        bsize = x.shape[0]
        z = tf.random.normal((bsize, self.z_dim), 0, 1)
        return {
            'disc': self._disc_loss(z, x).numpy(),
            'gen': self._gen_loss(z).numpy(),
            'latent': self._latent_loss(z).numpy(),
        }

    def _update(self, x, loss_fn, var, opt):
        """Update weights with gradient and optimizer.
        Args:
            x: tf.Tensor, [B, output_dim], output samples.
            loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
                loss function.
            var: List[tf.Tensor], trainable variables.
            opt: tf.keras.optimizers.Optimizer, keras optimizer.
        Returns:
            z: np.array, [B, z_dim], sampled latent prior.
            loss: np.array, [], loss value.
        """
        z = tf.random.normal((x.shape[0], self.z_dim), 0, 1)
        with tf.GradientTape() as tape:
            loss = loss_fn(z, x)
        
        grad = tape.gradient(loss, var)
        opt.apply_gradients(zip(grad, var))
        return z.numpy(), loss.numpy()

    def trainstep(self, x):
        """Optimize ALAE objective.
        Args:
            x: tf.Tensor, [B, output_dim], output samples.
        Returns:
            Dict[str, np.array], loss values.
        """
        _, dloss = self._update(x, self._disc_loss, self.ed_var, self.ed_opt)
        _, gloss = self._update(x, self._gen_loss, self.fg_var, self.fg_opt)
        _, lloss = self._update(x, self._latent_loss, self.eg_var, self.eg_opt)
        return {
            'disc': dloss,
            'gen': gloss,
            'latent': lloss,
        }

    @staticmethod
    def default_setting(z_dim=128, latent_dim=50, output_dim=784 + 10):
        """Default settings.
        """
        return {
            'z_dim': z_dim,
            'latent_dim': latent_dim,
            'output_dim': output_dim,
            'gamma': 10,
            'f': [1024, latent_dim],
            'g': [1024, output_dim],
            'e': [1024, latent_dim],
            'd': [1024, 1],
            'lr': 0.002,
            'beta1': 0.0,
            'beta2': 0.99,
        }
