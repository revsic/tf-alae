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

        self.f = tf.keras.Sequential([
            tf.keras.layers.Dense(dim) for dim in settings['f']])
        self.g = tf.keras.Sequential([
            tf.keras.layers.Dense(dim) for dim in settings['g']])
        self.e = tf.keras.Sequential([
            tf.keras.layers.Dense(dim) for dim in settings['e']])
        self.d = tf.keras.Sequential([
            tf.keras.layers.Dense(dim) for dim in settings['d']])

        self.f.build((None, self.z_dim))
        self.g.build((None, self.latent_dim))
        self.e.build((None, self.output_dim))
        self.d.build((None, self.latent_dim))

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
    def default_setting():
        return {
            'z_dim': 50,
            'latent_dim': 50,
            'output_dim': 784,
            'gamma': 1,
            'f': [1024, 50],
            'g': [1024, 784],
            'e': [1024, 50],
            'd': [1024, 1],
            'lr': 0.002,
            'beta1': 0.0,
            'beta2': 0.99,
        }
