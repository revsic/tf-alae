import sys
import os

import numpy as np
import tensorflow as tf

from utils.trainer import Trainer
from datasets.mnist import MNIST
from models.mlpalae import MlpAlae


def main(_):
    # set random seed
    tf.random.set_seed(1234)
    np.random.seed(1234)

    mnist = MNIST()
    mlpalae = MlpAlae()

    modelname = 'baseline'
    if not os.path.exists('./summary/' + modelname):
        os.mkdir('./summary/' + modelname)
    
    if not os.path.exists('./ckpt'):
        os.mkdir('./ckpt')

    trainer = Trainer('./summary/' + modelname, './ckpt/' + modelname)
    trainer.train(mlpalae, 50, mnist.cdatasets(), mnist.cdatasets(train=False))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
