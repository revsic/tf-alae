import os

import numpy as np
import tensorflow as tf
import tqdm


class Callback:
    """Callback object for customizing trainer.
    """
    def __init__(self):
        pass

    def interval(self):
        """Callback interval, -1 for epochs, positives for steps
        """
        raise NotImplementedError('Callback.interval is not implemented')

    def __call__(self, model, steps, epochs):
        """Call methods for manipulating models.
        Args:
            model: tf.keras.Model, model.
            steps: int, current steps.
            epochs: int, current epochs.
        """
        raise NotImplementedError('Callback.__call__ is not implemented')


class Trainer:
    """ALAE trainer.
    """
    def __init__(self,
                 summary_path,
                 ckpt_path,
                 ckpt_interval=None,
                 callback=None):
        """Initializer.
        Args:
            summary_path: str, path to the summary.
            ckpt_path: str, path to the checkpoint.
            ckpt_interval: int, write checkpoints in given steps.
            callback: Callback, callback object for customizing.
        """
        train_path = os.path.join(summary_path, 'train')
        test_path = os.path.join(summary_path, 'test')
        self.train_summary = tf.summary.create_file_writer(train_path)
        self.test_summary = tf.summary.create_file_writer(test_path)
        self.ckpt_path = ckpt_path
        self.ckpt_interval = ckpt_interval
        self.callback = callback

    def train(self, model, epochs, trainset, testset, trainlen=None):
        """Train ALAE model with given datasets.
        Args:
            model: ALAE, tf.keras.Model, ALAE model.
            epochs: int, the number of the epochs.
            trainset: tf.data.Dataset, training dataset.
            testset: tf.data.Dataset, test dataset.
            trainlen: int, length of the trainset.
        """
        step = 0
        cb_intval = self.callback.interval() \
            if self.callback is not None else None
        for epoch in tqdm.tqdm(range(epochs)):
            if cb_intval == -1:
                # run callback in epoch order
                self.callback(model, step, epoch)

            # training phase
            with tqdm.tqdm(total=trainlen, leave=False) as pbar:
                for datum in trainset:
                    step += 1
                    losses = model.trainstep(datum)
                    self.write_summary(losses, step)

                    if self.ckpt_interval is not None and \
                            step % self.ckpt_interval == 0:
                        # if training set is too large
                        _, flat = model(datum)
                        self.write_image(datum, flat, step, train=False)
                        model.save_weights(self.ckpt_path)
                    
                    if cb_intval is not None and \
                            cb_intval > 0 and step % cb_intval == 0:
                        # run callback in step order
                        self.callback(model, step, epoch)
                    pbar.update()

            _, flat = model(datum)
            self.write_image(datum, flat, step)

            # test phase
            metrics = []
            for datum in testset:
                metrics.append(model.losses(datum))

            self.write_summary(self.mean_metric(metrics), step, train=False)

            _, flat = model(datum)
            self.write_image(datum, flat, step, train=False)

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

    def write_image(self, datum, flat, step, train=True, name='image'):
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
            tf.summary.image(name + '_gt', datum[idx:idx + 1], step=step)

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
