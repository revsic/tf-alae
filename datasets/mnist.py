import numpy as np
import tensorflow as tf


class MNIST:
    """MNIST dataset wrapper.
    Attributes:
        x_train: np.ndarray, [B, 784(=28 * 28)], dataset for training.
        x_test: np.ndarray, [B, 784(=28 * 28)], dataset for testing.
        y_train: np.ndarray, [B], label for training, 0 ~ 9.
        y_test: np.ndarray, [B], label for testing, 0 ~ 9.
    """
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.x_train = x_train / 127.5 - 1.
        self.x_test = x_test / 127.5 - 1.
        self.y_train, self.y_test = y_train, y_test

    def rawdata(self, train=True):
        """Raw dataset pair.
        Args:
            train: bool, whether training mode or not.
        Returns:
            (np.ndarray, np.ndarray), [[B, 784], [B]],
                dataset and label pair.
        """
        return (self.x_train, self.y_train) \
            if train else (self.x_test, self.y_test)

    def datasets(self, bsize=128, bufsiz=10000, train=True):
        """Image dataset.
        Args:
            bsize: int, batch size.
            bufsiz: int, buffer size for shuffle.
            train: bool, whether training mode or not.
        Returns:
            tf.data.Dataset, tensorflow dataset object,
                Iterable[tf.Tensor=[B, 28, 28]], iterable.
        """
        x, _ = self.rawdata(train)
        return tf.data.Dataset.from_tensor_slices(x) \
            .shuffle(bufsiz) \
            .batch(bsize)

    def cdatasets(self, bsize=128, bufsiz=10000, train=True):
        """Conditioned dataset.
        Args:
            bsize: int, batch size.
            bufsiz: int, buffer size for shuffle.
            train: bool, whether training mode or not.
        Returns:
            tf.data.Dataset, tensorflow dataset object,
                Iterable[tf.Tensor=[B, 784 + 10]], iterable.
        """
        x, y = self.rawdata(train)
        data = np.concatenate([x.reshape(-1, 784), np.eye(10)[y]], axis=-1)
        return tf.data.Dataset.from_tensor_slices(data) \
            .shuffle(bufsiz) \
            .batch(bsize)
