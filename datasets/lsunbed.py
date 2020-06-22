import cv2
import lmdb
import numpy as np
import tensorflow as tf


class LsunBed:
    def __init__(self, path):
        self.path = path

    def reader(self):
        env = lmdb.open(self.path, map_size=1099511627776,
                        max_readers=100, readonly=True)
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for _, val in cursor:
                yield cv2.imdecode(
                    np.fromstring(val, dtype=np.uint8), 1)

    def datasets(self, bsize, bufsiz):
        return tf.data.Dataset.from_generator(
            self.reader, np.uint8)
