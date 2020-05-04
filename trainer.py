import tensorflow as tf

from mnist import MNIST
from mlpalae import MlpAlae


class Trainer:
    def __init__(self):
        pass

    def train(self, model, epochs, trainset, testset):
        step = 0
        for _ in range(epochs):
            for datum in trainset:
                step += 1
                losses = model.trainstep(datum)
                for key, value in losses.items():
                    tf.summary.scalar(key, value, step=step)


if __name__ == '__main__':
    mnist = MNIST()
    mlpalae = MlpAlae()

    trainer = Trainer()
    with tf.summary.create_file_writer('./summaries/train').as_default():
        trainer.train(mlpalae, 600, mnist.datasets(), mnist.datasets(train=False))
