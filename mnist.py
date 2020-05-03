import tensorflow as tf


class MNIST:
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (x_train, _), (x_test, _) = mnist.load_data()
        self.x_train, self.x_test = x_train / 255., x_test / 255.

    def rawdata(self, train=True):
        return self.x_train if train else self.x_test

    def datasets(self, bsize=128, train=True):
        return tf.data.Dataset.from_tensor_slices(self.rawdata(train)) \
            .shuffle(10000) \
            .batch(bsize)
