import tensorflow as tf


class ALAE(tf.keras.Model):
    """Interface for trainable ALAE object.
    (ALAE: Adversarial Latent Autoencoder, Pidhorskyi et al., 2020.)
    """
    def __init__(self):
        super(ALAE, self).__init__()

    def encoder(self, *args, **kwargs):
        """Encode data to fixed length latent vector.
        Returns:
            tf.Tensor, [B, ...], latent vectors.
        """
        raise NotImplementedError('ALAE.encoder is not implemented')

    def generator(self, *args, **kwargs):
        """Generate data from encoded latent.
        Returns:
            tf.Tensor, [B, ...], input tensors.
        """
        raise NotImplementedError('ALAE.generator is not implemented')

    def call(self, x):
        """Generate the latent vectors and autoencode the inputs.
        Args:
            x: tf.Tensor, [B, ...], input tensors.
        Returns:
            latent: tf.Tensor, [B, ...], latent vectors.
            _: tf.Tensor, [B, ...], autoencoded tensors.
        """
        latent = self.encoder(x)
        return latent, self.generator(latent)
    
    def losses(self, x):
        """Return loss values, discrimnator, generator, latent loss.
        Returns:
            Dict[str, np.ndarray], loss values.
        """
        raise NotImplementedError('ALAE.metrics is not implemented')

    def trainstep(self, x):
        """Optimize the model with given x.
        Returns:
            Dict[str, np.ndarray], loss values.
        """
        raise NotImplementedError('ALAE.trainstep is not implemented')
