import argparse
import os

import numpy as np
import tensorflow as tf

from utils.trainer import Callback, Trainer
from datasets.lsunbed import LsunBed
from models.stylealae import StyleAlae

PARSER = argparse.ArgumentParser()
PARSER.add_argument('option', type=str, default='train')
PARSER.add_argument('--name', default='style_lsunbed_lreq')
PARSER.add_argument('--summarydir', default='./summary')
PARSER.add_argument('--ckptdir', default='./ckpt')
PARSER.add_argument('--epochs', default=10, type=int)
PARSER.add_argument('--ckpt_interval', default=1000, type=int)
PARSER.add_argument('--seed', default=1234, type=int)
PARSER.add_argument('--dataset', default='D:\\bedroom_train_lmdb\\bedroom_train_lmdb')
PARSER.add_argument('--evalset', default='D:\\bedroom_val_lmdb\\bedroom_val_lmdb')

NUM_LAYERS = 7
RESOLUTION = (NUM_LAYERS + 1) ** 2
EPOCHS_PER_LEVEL = 1


class LevelController(Callback):
    """Training level controller.
    """
    def __init__(self,
                 num_layers=NUM_LAYERS,
                 resolution=RESOLUTION,
                 epochs_per_level=EPOCHS_PER_LEVEL):
        super(LevelController, self).__init__()
        self.num_layers = num_layers
        self.resolution = resolution
        self.epochs_per_level = epochs_per_level
    
    @override
    def interval(self):
        """Set callback interval as epoch.
        """
        return -1

    @override
    def __call__(self, model, _, epochs):
        """Set training level of models based on epochs.
        """
        level = min(self.num_layers - 1, epochs // self.epochs_per_level)
        model.set_level(level)


class StyleLsunBed(StyleAlae):
    """Style-ALAE for LSUN Bed dataset.
    """
    def __init__(self, settings=None):
        if settings is None:
            settings = StyleLsunBed.default_setting()

        super(StyleLsunBed, self).__init__(settings)

    def generate(self, z):
        """Generate output tensors from latent vectors.
            + denormalize and reshape to image.
        Args:
            _: tf.Tensor, [B, latent_dim], latent vectors.
        Returns:
            _: tf.Tensor, [B, 256, 256, 3], output tensors.
        """
        x = super().generate(z)
        return tf.clip_by_value(x, -1, 1)

    @staticmethod
    def default_setting():
        return {
            'latent_dim': 256,
            'num_layers': NUM_LAYERS,  # 7 => resolution 256x256
            'map_num_layers': 5,
            'disc_num_layers': 3,
            'init_channels': 8,
            'max_channels': 256,
            'out_channels': 3,
            'lr': 0.0015,
            'beta2': 0.99,
            'gamma': 10,
        }


"""
TODO: blending factor, pixel norm, lod driver (trainer callback, dynamic batch)
"""

def train(args):
    lsunbed = LsunBed(args.dataset)
    lsunbed_eval = LsunBed(args.evalset)
    stylealae = StyleLsunBed()

    modelname = args.name
    summary_path = os.path.join(args.summarydir, modelname)
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

    ckpt_path = os.path.join(args.ckptdir, modelname)
    if not os.path.exists(args.ckptdir):
        os.makedirs(args.ckptdir)

    trainer = Trainer(summary_path, ckpt_path, args.ckpt_interval)
    trainer.train(
        stylealae,
        args.epochs,
        lsunbed.datasets(),
        lsunbed_eval.datasets())

    return 0


def main(args):
    # set random seed
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    if args.option == 'train':
        train(args)


if __name__ == '__main__':
    main(PARSER.parse_args())
