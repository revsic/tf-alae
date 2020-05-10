import tensorflow as tf


class ALAE(tf.keras.Model):
    """Interface for trainable ALAE object.
    (ALAE: Adversarial Latent Autoencoder, Pidhorskyi et al., 2020.)
    """
    def __init__(self):
        super(ALAE, self).__init__()

    def prepare(self, z_dim, gamma, learning_rate, beta1, beta2):
        """Prepare for training and inference.
        """
        self.z_dim = z_dim
        self.gamma = gamma

        self.map = self.mapper()
        self.gen = self.generator()
        self.enc = self.encoder()
        self.dec = self.discriminator()

        self.fakepass = tf.keras.Sequential([
            self.map, self.gen, self.enc, self.dec])
        self.realpass = tf.keras.Sequential([self.enc, self.dec])
        self.latentpass = tf.keras.Sequential([self.map, self.gen, self.enc])

        self.ed_var = self.enc.trainable_variables + self.dec.trainable_variables
        self.fg_var = self.map.trainable_variables + self.gen.trainable_variables
        self.eg_var = self.enc.trainable_variables + self.gen.trainable_variables

        self.ed_opt = tf.keras.optimizers.Adam(learning_rate, beta1, beta2)
        self.fg_opt = tf.keras.optimizers.Adam(learning_rate, beta1, beta2)
        self.eg_opt = tf.keras.optimizers.Adam(learning_rate, beta1, beta2)

    def encode(self, *args, **kwargs):
        """Encode the input tensors to latent vectors.
        Args:
            _: tf.Tensor, [B, ...], input tensors.
        Returns:
            _: tf.Tensor, [B, latent_dim], latent vectors.
        """
        return self.enc(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """Generate output tensors from latent vectors.
        Args:
            _: tf.Tensor, [B, latent_dim], latent vectors.
        Returns:
            _: tf.Tensor, [B, ...], output tensors.
        """
        return self.gen(*args, **kwargs)

    def call(self, x):
        """Generate the latent vectors and autoencode the inputs.
        Args:
            x: tf.Tensor, [B, ...], input tensors.
        Returns:
            latent: tf.Tensor, [B, latent_dim], latent vectors.
            _: tf.Tensor, [B, ...], autoencoded tensors.
        """
        latent = self.encode(x)
        return latent, self.generate(latent)

    @tf.function
    def _disc_loss(self, z, x):
        """Compute discriminator loss.
        Args:
            z: tf.Tensor, [B, z_dim], latent prior.
            x: tf.Tensor, [B, ...], output tensors.
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
            x: tf.Tensor, [B, ...], output samples.
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
            x: tf.Tensor, [B, ...], output samples.
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
            x: tf.Tensor, [B, ...], output samples.
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

    def mapper(self, *args, **kwargs):
        """Model for mpping latent from prior.
        Returns:
            tf.keras.Model: map prior to latent.
        """
        raise NotImplementedError('ALAE.mapper is not implemented')

    def generator(self, *args, **kwargs):
        """Model for generating data from encoded latent.
        Returns:
            tf.keras.Model: generate sample from encoded latent.
        """
        raise NotImplementedError('ALAE.generator is not implemented')

    def encoder(self, *args, **kwargs):
        """Model for encoding data to fixed length latent vector.
        Returns:
            tf.keras.Model: encode data to fixed length latent vector.
        """
        raise NotImplementedError('ALAE.encoder is not implemented')

    def discriminator(self, *args, **kwargs):
        """Model for discriminating real sample from fake one.
        Returns:
            tf.keras.Model: discriminate real sample from fake one.
        """
        raise NotImplementedError('ALAE.discriminator is not implemented')
