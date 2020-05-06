import os

import numpy as np
import tensorflow as tf
import tqdm

from mnist import MNIST
from mlpalae import MlpAlae


class Trainer:
    def __init__(self, summary_path, ckpt_path):
        train_path = os.path.join(summary_path, 'train')
        test_path = os.path.join(summary_path, 'test')
        self.train_summary = tf.summary.create_file_writer(train_path)
        self.test_summary = tf.summary.create_file_writer(test_path)
        self.ckpt_path = ckpt_path

    def train(self, model, epochs, trainset, testset):
        step = 0
        for _ in tqdm.tqdm(range(epochs)):
            for datum in trainset:
                step += 1
                datum = tf.reshape(datum, (datum.shape[0], -1))
                losses = model.trainstep(datum)
                self.write_summary(losses, step)

            _, flat = model(datum)
            self.write_image(flat, step)

            metrics = []
            for datum in testset:
                datum = tf.reshape(datum, (datum.shape[0], -1))
                metrics.append(model.losses(datum))

            self.write_summary(self.mean_metric(metrics), step, train=False)

            _, flat = model(datum)
            self.write_image(flat, step, train=False)

            model.save_weights(self.ckpt_path)

    def write_summary(self, metrics, step, train=True):
        summary = self.train_summary if train else self.test_summary
        with summary.as_default():
            for key, value in metrics.items():
                tf.summary.scalar(key, value, step=step)

    def write_image(self, flat, step, train=True, name='image'):
        idx = np.random.randint(flat.shape[0])
        image = tf.clip_by_value(flat[idx], -1, 1)
        summary = self.train_summary if train else self.test_summary
        with summary.as_default():
            tf.summary.image(
                name,
                tf.reshape(image[:784], (1, 28, 28, 1)),
                step=step)
    
    def mean_metric(self, metrics):
        size = 0
        avgs = {}
        for metric in metrics:
            size += 1
            for key, value in metric.items():
                if key not in avgs:
                    avgs[key] = 0
                avgs[key] += value
        
        for key in avgs.keys():
            avgs[key] /= size

        return avgs


if __name__ == '__main__':
    mnist = MNIST()
    mlpalae = MlpAlae()

    modelname = 'lat50_gam100_sepopt'
    if not os.path.exists('./summary/' + modelname):
        os.mkdir('./summary/' + modelname)
    
    if not os.path.exists('./ckpt'):
        os.mkdir('./ckpt')

    trainer = Trainer('./summary/' + modelname, './ckpt/' + modelname)
    trainer.train(mlpalae, 600, mnist.cdatasets(), mnist.cdatasets(train=False))
