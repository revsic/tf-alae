import os

import tensorflow as tf
import tqdm

from mnist import MNIST
from mlpalae import MlpAlae


class Trainer:
    def __init__(self, summary_path):
        train_path = os.path.join(summary_path, 'train')
        test_path = os.path.join(summary_path, 'test')
        self.train_summary = tf.summary.create_file_writer(train_path)
        self.test_summary = tf.summary.create_file_writer(test_path)

    def train(self, model, epochs, trainset, testset):
        step = 0
        for _ in tqdm.tqdm(range(epochs)):
            for datum in trainset:
                step += 1
                losses = model.trainstep(datum)
                self.write_summary(losses, step)
            
            for datum in testset:
                self.write_summary(model.losses(datum), step, train=False)

    def write_summary(self, metrics, step, train=True):
        summary = self.train_summary if train else self.test_summary
        with summary.as_default():
            for key, value in metrics.items():
                tf.summary.scalar(key, value, step=step)


if __name__ == '__main__':
    mnist = MNIST()
    mlpalae = MlpAlae()

    trainer = Trainer('./summary')
    trainer.train(mlpalae, 600, mnist.datasets(), mnist.datasets(train=False))
