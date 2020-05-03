import tensorflow as tf


class MlpALAE(tf.keras.Model):
    def __init__(self, settings=None):
        super(MlpALAE, self).__init__()
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
        self.fakelatent = tf.keras.Sequential([self.f, self.g, self.e])

        self.ed_var = self.e.trainable_variables + self.d.trainable_variables
        self.fg_var = self.f.trainable_variables + self.g.trainable_variables
        self.eg_var = self.e.trainable_variables + self.g.trainable_variables

        self.optimizer = tf.keras.optimizers.Adam(
            settings['lr'], settings['beta1'], settings['beta2'])

    @tf.function
    def trainstep(self, x):
        bsize = x.shape[0]
        z = tf.random.normal((bsize, self.z_dim), 0, 1)
        with tf.GradientTape() as tape, tf.GradientTape() as regtape:
            fake = self.fakepass(z)
            real = self.realpass(x)

            grad = regtape.gradient(real, self.ed_var)
            reg = self.gamma / 2 * tf.reduce_mean(
                [tf.reduce_mean(tf.square(x)) for x in grad])

            fakeloss = tf.reduce_mean(tf.math.softplus(fake))
            realloss = tf.reduce_mean(tf.math.softplus(-real))
            loss = fakeloss + realloss + reg
        
        grad = tape.gradient(loss, self.ed_var)
        self.optimizer.apply_gradients(zip(grad, self.ed_var))

        z = tf.random.normal((bsize, self.z_dim), 0, 1)
        with tf.GradientTape() as tape:
            fake = self.fakepass(z)
            fakeloss = tf.reduce_mean(tf.math.softplus(-fake))
        
        grad = tape.gradient(fakeloss, self.fg_var)
        self.optimizer.apply_gradients(zip(grad, self.fg_var))

        z = tf.random.normal((bsize, self.z_dim), 0, 1)
        with tf.GradientTape() as tape:
            latent = self.f(z)
            fakelatent = self.fakelatent(z)
            loss = tf.reduce_mean(tf.square(latent - fakelatent))
        
        grad = tape.gradient(loss, self.eg_var)
        self.optimizer.apply_gradients(zip(grad, self.eg_var))

    @staticmethod
    def default_setting(z_dim=128, latent_dim=50, output_dim=784):
        return {
            'z_dim': z_dim,
            'latent_dim': latent_dim,
            'output_dim': output_dim,
            'gamma': 1,
            'f': [1024, latent_dim],
            'g': [1024, output_dim],
            'e': [1024, latent_dim],
            'd': [1024, 1],
            'lr': 0.002,
            'beta1': 0.9,
            'beta2': 0.99,
        }
