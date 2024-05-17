# LGE-CNN: Lattice Gauge Equivariant Convolutional Neural Networks
(formerly known as "cloughmore")

This is the code repository for the paper "Lattice Gauge Equivariant Convolutional Neural Networks" ([arXiv:2012:12901](https://arxiv.org/abs/2012.12901))
by M. Favoni, A. Ipp, D. I. Müller and D. Schuh.  Our code includes a basic SU(2) Yang-Mills code that is used to
generate datasets found in the package `lge_cnn.ym`. The machine learning code (model classes, layers, datasets classes)
is in `lge_cnn.nn`.

## Conda environment

```shell
conda env create -f environment.yml
```

## Generating datasets

Datasets can be generated using the ``generate_dataset.py`` script in the ``scripts`` folder.

To generate the datasets used in the paper, run the shell scripts
```shell
./scripts/generate_2d.sh
./scripts/generate_4d.sh
```

## Training

Training scripts are in the ``scripts`` directory. 

To train all 1+1D models used in the paper, run the shell scripts
```shell
./scripts/train_D2_trW_1.sh
./scripts/train_D2_trW_1_base.sh
./scripts/train_D2_trW_1x2.sh
./scripts/train_D2_trW_1x2_base.sh
./scripts/train_D2_trW_2.sh
./scripts/train_D2_trW_2_base.sh
./scripts/train_D2_trW_4.sh
./scripts/train_D2_trW_4_base.sh
```

For 3+1D models, run
```shell
./scripts/train_D4_trW_2.sh
./scripts/train_D4_trW_4.sh
./scripts/train_D4_QP.sh
```

## Testing

Tests and plot data is generated using the  ``Test all results.ipynb`` Jupyter notebook in the ``notebooks`` directory.
