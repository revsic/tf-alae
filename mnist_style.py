import argparse
import os

import numpy as np
import tensorflow as tf

from utils.trainer import Trainer
from datasets.mnist import MNIST
from models.stylealae import StyleAlae

PARSER = argparse.ArgumentParser()
PARSER.add_argument('option', type=str, default='train')
PARSER.add_argument('--name', default='style_mnist')
PARSER.add_argument('--summarydir', default='./summary')
PARSER.add_argument('--ckptdir', default='./ckpt')
PARSER.add_argument('--epochs', default=50, type=int)
PARSER.add_argument('--seed', default=1234, type=int)


class StyleMNIST(StyleAlae):
    """MLP-ALAE for MNIST dataset.
    """
    def __init__(self, settings=None):
        super(StyleMNIST, self).__init__(settings)

    def generate(self, z):
        """Generate output tensors from latent vectors.
            + denormalize and reshape to image.
        Args:
            _: tf.Tensor, [B, latent_dim], latent vectors.
        Returns:
            _: tf.Tensor, [B, 32, 32, 1], output tensors.
        """
        x = super().generate(z)
        return tf.clip_by_value(x, -1, 1)


def train(args):
    mnist = MNIST()
    mlpalae = StyleAlae()

    modelname = args.name
    summary_path = os.path.join(args.summarydir, modelname)
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    
    ckpt_path = os.path.join(args.ckptdir, modelname)
    if not os.path.exists(args.ckptdir):
        os.makedirs(args.ckptdir)

    trainer = Trainer(summary_path, ckpt_path)
    trainer.train(
        mlpalae,
        args.epochs,
        mnist.datasets(padding=2, flatten=False),
        mnist.datasets(padding=2, flatten=False, train=False))

    return 0


def main(args):
    # set random seed
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    if args.option == 'train':
        train(args)


if __name__ == '__main__':
    main(PARSER.parse_args())