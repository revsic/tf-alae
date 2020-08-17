import argparse
import os

import numpy as np
import tensorflow as tf

from utils.trainer import Trainer
from datasets.mnist import MNIST
from models.mlpalae import MlpAlae

PARSER = argparse.ArgumentParser()
PARSER.add_argument('option', type=str, default='train')
PARSER.add_argument('--name', default='mnist_mlp')
PARSER.add_argument('--summarydir', default='./summary')
PARSER.add_argument('--ckptdir', default='./ckpt')
PARSER.add_argument('--epochs', default=50, type=int)
PARSER.add_argument('--seed', default=1234, type=int)
PARSER.add_argument('--batch-size', default=128, type=int)


class MnistAlae(MlpAlae):
    """MLP-ALAE for MNIST dataset (flatten, conditioned).
    """
    def __init__(self, settings=None):
        if settings is None:
            settings = MnistAlae.default_setting()
        super(MnistAlae, self).__init__(settings)

    def generate(self, z):
        """Generate output tensors from latent vectors.
            + denormalize and reshape to image.
        Args:
            _: tf.Tensor, [B, latent_dim], latent vectors.
        Returns:
            _: tf.Tensor, [B, ...], output tensors.
        """
        x = super().generate(z)
        x = tf.clip_by_value(x[:, :784], 0, 1)
        return tf.reshape(x, [-1, 28, 28, 1])

    @staticmethod
    def default_setting(z_dim=128, latent_dim=50, output_dim=784 + 10):
        """Default settings.
        """
        return {
            'z_dim': z_dim,
            'latent_dim': latent_dim,
            'output_dim': output_dim,
            'gamma': 10,
            'f': [1024, latent_dim],
            'g': [1024, output_dim],
            'e': [1024, latent_dim],
            'd': [1024, 1],
            'lr': 0.002,
            'beta1': 0.0,
            'beta2': 0.99,
        }


def train(args):
    mnist = MNIST()
    mlpalae = MnistAlae()

    modelname = args.name
    summary_path = os.path.join(args.summarydir, modelname)
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    
    ckpt_path = os.path.join(args.ckptdir, modelname)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    trainer = Trainer(summary_path, ckpt_path)
    trainer.train(
        mlpalae,
        args.epochs,
        mnist.datasets(bsize=args.batch_size, flatten=True, condition=True),
        mnist.datasets(bsize=args.batch_size, flatten=True, condition=True, train=False),
        len(mnist.x_train) // args.batch_size)

    return 0


def main(args):
    # set random seed
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    if args.option == 'train':
        train(args)


if __name__ == '__main__':
    main(PARSER.parse_args())
