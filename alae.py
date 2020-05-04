import tensorflow as tf


class ALAE(tf.keras.Model):
    def __init__(self):
        super(ALAE, self).__init__()

    def encoder(self, *args, **kwargs):
        raise NotImplementedError('ALAE.encoder is not implemented')

    def generator(self, *args, **kwargs):
        raise NotImplementedError('ALAE.generator is not implemented')

    def call(self, x):
        latent = self.encoder(x)
        return latent, self.generator(latent)
    
    def losses(self, x):
        raise NotImplementedError('ALAE.metrics is not implemented')

    def trainstep(self, x):
        raise NotImplementedError('ALAE.trainstep is not implemented')
