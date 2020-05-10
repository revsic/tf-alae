import argparse
import os

import numpy as np
import tensorflow as tf

from utils.trainer import Trainer
from datasets.mnist import MNIST
from models.mlpalae import MlpAlae

PARSER = argparse.ArgumentParser()
PARSER.add_argument('option', type=str, default='train')
PARSER.add_argument('--name', default='baseline')
PARSER.add_argument('--summarydir', default='./summary')
PARSER.add_argument('--ckptdir', default='./ckpt')
PARSER.add_argument('--epochs', default=50, type=int)
PARSER.add_argument('--seed', default=1234, type=int)


def train(args):
    mnist = MNIST()
    mlpalae = MlpAlae()

    modelname = args.name
    summary_path = os.path.join(args.summarydir, modelname)
    if not os.path.exists(summary_path):
        os.mkdir(summary_path)
    
    ckpt_path = os.path.join(args.ckptdir, modelname)
    if not os.path.exists(args.ckptdir):
        os.mkdir(args.ckptdir)

    trainer = Trainer(summary_path, ckpt_path)
    trainer.train(mlpalae, args.epochs, mnist.cdatasets(), mnist.cdatasets(train=False))
    return 0


def main(args):
    # set random seed
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    if args.option == 'train':
        train(args)


if __name__ == '__main__':
    main(PARSER.parse_args())
