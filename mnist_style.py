import argparse
import logging
import os

import numpy as np
import tensorflow as tf

from utils.trainer import Callback, Trainer
from datasets.mnist import MNIST
from models.stylealae import StyleAlae

PARSER = argparse.ArgumentParser()
PARSER.add_argument('option', type=str, default='train')
PARSER.add_argument('--name', default='style_mnist')
PARSER.add_argument('--summarydir', default='./summary')
PARSER.add_argument('--ckptdir', default='./ckpt')
PARSER.add_argument('--epochs', default=10, type=int)
PARSER.add_argument('--seed', default=1234, type=int)
PARSER.add_argument('--batch_size', default=128, type=int)

NUM_LAYERS = 4
RESOLUTION = (NUM_LAYERS + 1) ** 2
EPOCHS_PER_LEVEL = 2


class LevelController(Callback):
    """Training level controller.
    """
    def __init__(self,
                 num_layers=NUM_LAYERS,
                 epochs_per_level=EPOCHS_PER_LEVEL):
        super(LevelController, self).__init__()
        self.num_layers = num_layers
        self.epochs_per_level = epochs_per_level

    def interval(self):
        """Set callback interval as epoch.
        """
        return -1

    def __call__(self, model, _, epochs):
        """Set training level of models based on epochs.
        """
        level = min(self.num_layers - 1, epochs // self.epochs_per_level)
        model.set_level(level)


class StyleMNIST(StyleAlae):
    """Style-ALAE for MNIST dataset.
    """
    def __init__(self, settings=None):
        if settings is None:
            settings = StyleMNIST.default_setting()

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

    @staticmethod
    def default_setting():
        return {
            'latent_dim': 50,
            'num_layers': NUM_LAYERS,
            'map_num_layers': 3,
            'disc_num_layers': 3,
            'init_channels': 4,
            'max_channels': 256,
            'out_channels': 1,
            'lr': 1e-4,
            'beta1': 0.9,
            'beta2': 0.99,
            'gamma': 10,
        }


def train(args):
    tf.get_logger().setLevel(logging.ERROR)

    mnist = MNIST()
    stylealae = StyleMNIST()

    modelname = args.name
    summary_path = os.path.join(args.summarydir, modelname)
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    
    ckpt_path = os.path.join(args.ckptdir, modelname)
    if not os.path.exists(args.ckptdir):
        os.makedirs(args.ckptdir)

    trainer = Trainer(summary_path, ckpt_path, callback=LevelController())
    trainer.train(
        stylealae,
        args.epochs,
        mnist.datasets(args.batch_size, padding=2, flatten=False),
        mnist.datasets(args.batch_size, padding=2, flatten=False, train=False),
        trainlen=len(mnist.x_train) // args.batch_size)

    return 0


def main(args):
    # set random seed
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    if args.option == 'train':
        train(args)


if __name__ == '__main__':
    main(PARSER.parse_args())
