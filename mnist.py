import numpy as np
import tensorflow as tf


class MNIST:
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.x_train = x_train / 127.5 - 1.
        self.x_test = x_test / 127.5 - 1.
        self.y_train, self.y_test = y_train, y_test

    def rawdata(self, train=True):
        return (self.x_train, self.y_train) \
            if train else (self.x_test, self.y_test)

    def datasets(self, bsize=128, bufsiz=10000, train=True):
        x, _ = self.rawdata(train)
        return tf.data.Dataset.from_tensor_slices(x) \
            .shuffle(bufsiz) \
            .batch(bsize)

    def cond_datasets(self, bsize=128, bufsiz=10000, train=True):
        x, y = self.rawdata(train)
        data = np.concatenate([x.reshape(-1, 784), np.eye(10)[y]], axis=-1)
        return tf.data.Dataset.from_tensor_slices(x) \
            .shuffle(bufsiz) \
            .batch(bsize)
