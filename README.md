# Collapse in continuous vs discrete latent spaces

Analyzing latent collapse in variational auto-encoder models: VAE, VQ-VAE, and RQ-VAE on MNIST and ABO.

## What's included

- Models: VAE, VQ-VAE, RQ-VAE (encoder-decoder with configurable channels and latent size).
- Data: MNIST (28x28 grayscale) and ABO (64x64 RGB). Loaders in `data/`.
- Training: `training/train.py` trains a given model on a dataset, logs reconstruction/KL (and VQ losses), active latent dims, and optionally FID/IS via wandb.
- Config: `config.py` defines defaults per dataset (MNIST vs ABO) and exposes `get_mnist_config()` and `get_abo_config()`.

## How to run

1. Install dependencies: `pip install -r requirements.txt`
2. From the project root (`ECE285FinalProject/`), run:
  - MNIST: `python run_mnist.py` (trains VAE and VQ-VAE with several decoder architectures and beta values).
  - ABO: `python run_abo.py` (same idea for the ABO dataset).

Training writes logs and images under `output/` (or the path set in config).

## Changing default configuration

Edit `config.py` for dataset defaults. To override at run time, pass kwargs into the config getters in the run scripts (e.g. `get_mnist_config(beta=0.5, epochs=100)`).