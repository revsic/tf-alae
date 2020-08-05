# tf-alae

(Unofficial) Tensorflow implementation of Adversarial Latent Autoencoder (ALAE, Pidhorskyi et al., 2020)

## Usage

To train the mnist model
```
python mnist_mlp.py train
```

To open tensorboard summary
```
tensorboard --logdir summary
```

Currently, StyleALAE is experimental.

```bash
# to train mnist
python mnist_style.py train

# to train lsunbed
python lsunbed_style.py train
```

To use released checkpoints, download files from [release](https://github.com/revsic/tf-alae/releases) and unzip it.

Following is example of [MNIST-MLP](https://github.com/revsic/tf-alae/releases/tag/MnistMlp).
```py
import json
from mnist_mlp import MnistAlae

with open('settings.json') as f:
    settings = json.load(f)

alae = MnistAlae(settings)
alae.load_weights('./mnist_mlp/mnist_mlp')
```

## Jupyter notebook

- [mnist_expr.ipynb](./experiments/mnist_expr.ipynb): MNIST interpolation with Mlp-ALAE

## Learning Curve

Mlp-ALAE + MNIST

![mnist mlp learning curve](rsrc/mnist_mlp.jpg)

## Sample

MNIST 0 ~ 4 polymorph

![mnist polymorph](rsrc/mnist_polymorph.png)
