import os

import numpy as np
import tensorflow as tf
import tqdm


class Trainer:
    """ALAE trainer.
    """
    def __init__(self, summary_path, ckpt_path, ckpt_interval=None):
        train_path = os.path.join(summary_path, 'train')
        test_path = os.path.join(summary_path, 'test')
        self.train_summary = tf.summary.create_file_writer(train_path)
        self.test_summary = tf.summary.create_file_writer(test_path)
        self.ckpt_path = ckpt_path
        self.ckpt_interval = ckpt_interval

    def train(self, model, epochs, trainset, testset):
        """Train ALAE model with given datasets.
        Args:
            model: ALAE, tf.keras.Model, ALAE model.
            epochs: int, the number of the epochs.
            trainset: tf.data.Dataset, training dataset.
            testset: tf.data.Dataset, test dataset.
        """
        step = 0
        for _ in range(epochs):
            # training phase
            for datum in tqdm.tqdm(trainset):
                step += 1
                losses = model.trainstep(datum)
                self.write_summary(losses, step)

                if self.ckpt_interval is not None and \
                        step % self.ckpt_interval == 0:
                    # if training set is too large
                    _, flat = model(datum)
                    self.write_image(flat, step, train=False)
                    model.save_weights(self.ckpt_path)

            _, flat = model(datum)
            self.write_image(flat, step)

            # test phase
            metrics = []
            for datum in testset:
                metrics.append(model.losses(datum))

            self.write_summary(self.mean_metric(metrics), step, train=False)

            _, flat = model(datum)
            self.write_image(flat, step, train=False)

            # write checkpoint
            model.save_weights(self.ckpt_path)

    def write_summary(self, metrics, step, train=True):
        """Write tensorboard summary.
        Args:
            metrics: Dict[str, np.array], metrics.
            step: int, current step.
            train: bool, whether in training phase or test phase.
        """
        summary = self.train_summary if train else self.test_summary
        with summary.as_default():
            for key, value in metrics.items():
                tf.summary.scalar(key, value, step=step)

    def write_image(self, flat, step, train=True, name='image'):
        """Write image to the tensorboard summary.
        Args:
            flat: tf.Tensor, [B, ...], autoencoded image.
            step: int, current step.
            train: bool, whether in training phase or test phase.
            name: str, image name for tensorboard.
        """
        # random index
        idx = np.random.randint(flat.shape[0])
        summary = self.train_summary if train else self.test_summary
        with summary.as_default():
            # write tensorboard summary
            tf.summary.image(name, flat[idx:idx + 1], step=step)

    def mean_metric(self, metrics):
        """Compute mean of the metrics.
        Args:
            List[Dict[str, np.array]], metrics.
        Returns:
            Dict[str, np.array], mean metrics.
        """
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
