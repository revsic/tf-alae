import tensorflow as tf

from alae import ALAE


class MlpAlae(ALAE):
    def __init__(self, settings=None):
        super(MlpAlae, self).__init__()
        if settings is None:
            settings = self.default_setting()

        self.z_dim = settings['z_dim']
        self.latent_dim = settings['latent_dim']
        self.output_dim = settings['output_dim']

        self.gamma = settings['gamma']

        def seq(dims, input_dim):
            layer = tf.keras.Sequential([
                tf.keras.layers.Dense(dim, activation='relu')
                for dim in dims[:-1]])
            layer.add(tf.keras.layers.Dense(dims[-1]))
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

        self.optimizer = tf.keras.optimizers.Adam(
            settings['lr'], settings['beta1'], settings['beta2'])

    def encoder(self, *args, **kwargs):
        return self.e(*args, **kwargs)

    def generator(self, *args, **kwargs):
        return self.g(*args, **kwargs)

    @tf.function
    def _disc_loss(self, z, x):
        with tf.GradientTape() as tape:
            fakeloss = tf.reduce_mean(tf.math.softplus(self.fakepass(z)))
            realloss = tf.reduce_mean(tf.math.softplus(-self.realpass(x)))

        grad = tape.gradient(realloss, self.ed_var)
        gradreg = self.gamma / 2 * tf.reduce_mean([
            tf.reduce_mean(tf.square(x)) for x in grad])

        return fakeloss + realloss + gradreg

    @tf.function
    def _gen_loss(self, z, _=None):
        return tf.reduce_mean(tf.math.softplus(-self.fakepass(z)))

    @tf.function
    def _latent_loss(self, z, _=None):
        latent = self.f(z)
        recovered = self.latentpass(z)
        return tf.reduce_mean(tf.square(latent - recovered))

    def losses(self, x):
        bsize = x.shape[0]
        z = tf.random.normal((bsize, self.z_dim), 0, 1)
        return {
            'disc': self._disc_loss(z, x),
            'gen': self._gen_loss(z),
            'latent': self._latent_loss(z),
        }

    def _update(self, x, loss_fn, var):
        z = tf.random.normal((x.shape[0], self.z_dim), 0, 1)
        with tf.GradientTape() as tape:
            loss = loss_fn(z, x)
        
        grad = tape.gradient(loss, var)
        self.optimizer.apply_gradients(zip(grad, var))
        return z.numpy(), loss.numpy()

    def trainstep(self, x):
        _, dloss = self._update(x, self._disc_loss, self.ed_var)
        _, gloss = self._update(x, self._gen_loss, self.fg_var)
        _, lloss = self._update(x, self._latent_loss, self.eg_var)
        return {
            'disc': dloss,
            'gen': gloss,
            'latent': lloss,
        }

    @staticmethod
    def default_setting(z_dim=128, latent_dim=50, output_dim=784):
        return {
            'z_dim': z_dim,
            'latent_dim': latent_dim,
            'output_dim': output_dim,
            'gamma': 1,
            'f': [1024, 1024, latent_dim],
            'g': [1024, 1024, output_dim],
            'e': [1024, 1024, latent_dim],
            'd': [1024, 1],
            'lr': 0.002,
            'beta1': 0.0,
            'beta2': 0.99,
        }
