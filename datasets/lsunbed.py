import cv2
import lmdb
import numpy as np
import tensorflow as tf


class LsunBed:
    """LSUN Bed dataset wrapper.
    Attributes:
        path: str, path of the dataset.
        size: int, image size.
        count: int, total samples of the dataset.
    """
    def __init__(self, path, size=256):
        self.path = path
        self.size = size

        env = lmdb.open(self.path)
        with env.begin(write=False) as txn:
            self.count = txn.stat()['entries']

    def reader(self):
        """Generator of reading images and normalizing for training.
        Yields:
            np.ndarray, [size, size, 3], normalized image in range (-1, 1).
        """
        env = lmdb.open(self.path, map_size=1099511627776,
                        max_readers=100, readonly=True)
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for _, val in cursor:
                # read image
                img = cv2.imdecode(
                    np.fromstring(val, dtype=np.uint8), 1)
                # resize
                img = LsunBed.center_crop(img, self.size, self.size)
                # normalize
                yield LsunBed.normalize(img)

    @staticmethod
    def center_crop(img, h, w):
        """Crop center of the image.
        Args:
            img: np.ndarray, [H, W, C], image array.
            h: int, height of the output.
            w: int, width of the output.
        Returns:
            np.ndarray, [h, w, C], cropped image.
        """
        height, width, _ = img.shape
        dh = (height - h) // 2
        dw = (width - w) // 2
        return img[dh:dh + h, dw:dw + w, :]
    
    @staticmethod
    def normalize(img):
        """Normalize image in range(-1, 1).
        Args:
            img: np.ndarray, image array.
        Returns:
            np.ndarray, normalized images.
        """
        return img / 127.5 - 1.

    def datasets(self, bsize=8, bufsiz=10000):
        """Image dataset.
        Args:
            bsize: int, batch size.
            bufsiz: int, buffer size for shuffling.
        Returns:
            tf.data.Dataset, tensorflow dataset object,
                Iterable[tf.Tensor=[B, size, size, 3]], iterable.
        """
        return tf.data.Dataset.from_generator(self.reader, np.float32) \
            .shuffle(bufsiz) \
            .batch(bsize)
