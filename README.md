# tf-alae

(Unofficial) Tensorflow implementation of Adversarial Latent Autoencoder (ALAE, Pidhorskyi et al., 2020)

## Usage

To train the mnist model
```
python trainer.py
```

To open tensorboard summary
```
tensorboard --logdir summary
```

Jupyter notebook
- [mnist_expr.ipynb](./mnist_expr.ipynb): MNIST experiments.

## Structure

- [alae.py](./alae.py): Abstracted ALAE object.
- [mlpalae.py](./mlpalae.py): MLP-ALAE implementation.
- [trainer.py](./trainer.py): MLP-ALAE trainer.
- [mnist.py](./mnist.py): MNIST wrapper.

## Autoencoded

![enumeration](./rsrc/enum.png)

## Polymorph

- two - three

![polymorph number two to three](./rsrc/two2three.png)
