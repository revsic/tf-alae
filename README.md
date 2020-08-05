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

## Jupyter notebook

- [mnist_expr.ipynb](./experiments/mnist_expr.ipynb): MNIST interpolation with Mlp-ALAE

## Learning Curve

Mlp-ALAE + MNIST

<img src="./rsrc/mnist_mlp_disc.jpg" width="30%">
<img src="./rsrc/mnist_mlp_gen.jpg" width="30%">
<img src="./rsrc/mnist_mlp_latent.jpg" width="30%">

## Sample

two to three

![two to three](./rsrc/two2three.png)

three to six

![three to six](./rsrc/three2six.png)

eight to nine

![eight to nine](./rsrc/eight2nine.png)
