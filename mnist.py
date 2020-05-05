import tensorflow as tf


class MNIST:
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (x_train, _), (x_test, _) = mnist.load_data()
        self.x_train = x_train / 127.5 - 1.
        self.x_test = x_test / 127.5 - 1.

    def rawdata(self, train=True):
        return self.x_train if train else self.x_test

    def datasets(self, bsize=128, bufsiz=10000, train=True):
        return tf.data.Dataset.from_tensor_slices(self.rawdata(train)) \
            .shuffle(bufsiz) \
            .batch(bsize)
