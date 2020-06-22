import cv2
import lmdb
import numpy as np
import tensorflow as tf


class LsunBed:
    def __init__(self, path, size=256):
        self.path = path
        self.size = size

    def reader(self):
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
        """
        Args:
            img: np.ndarray, [H, W, C]
            h: int, height.
            w: int, width.
        Returns:
            np.ndarray, [h, w, C]
        """
        height, width, _ = img.shape
        dh = (height - h) // 2
        dw = (width - w) // 2
        return img[dh:dh + h, dw:dw + w, :]
    
    @staticmethod
    def normalize(img):
        return img / 127.5 - 1.

    @staticmethod
    def denormalize(img):
        return (img + 1) * 127.5

    def datasets(self, bsize=128, bufsiz=10000):
        return tf.data.Dataset.from_generator(self.reader, np.float32) \
            .shuffle(bufsiz) \
            .batch(bsize)
