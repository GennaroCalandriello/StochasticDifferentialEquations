#!/bin/bash
export PYTHONWARNINGS="ignore"
n_train=1000
n_testval=100
n_sweep=100
n_warmup=2000

echo $n_train

# 4x8^3 dataset (training, validation and test)
mkdir "/media/data/dmueller/ym_datasets/D4_4x8p3/"

python generate_dataset.py \
        --paths "/media/data/dmueller/ym_datasets/D4_4x8p3/train.hdf5" \
        "/media/data/dmueller/ym_datasets/D4_4x8p3/val.hdf5" \
        "/media/data/dmueller/ym_datasets/D4_4x8p3/test.hdf5" \
        --nums $n_train $n_testval $n_testval \
        --sweeps $n_sweep \
        --warmup $n_warmup \
        --beta_min 0.1 \
        --beta_max 6.0 \
        --beta_steps 10 \
        --dims 4 8 8 8 \
        --loops 2 4 \
        --loop_axes 1 2 \
        --polyakov \
        --charge_plaq \
        --charge_clov

# 6x8^3 dataset (test)
mkdir "/media/data/dmueller/ym_datasets/D4_6x8p3/"

python generate_dataset.py \
        --paths "/media/data/dmueller/ym_datasets/D4_6x8p3/test.hdf5" \
        --nums $n_testval \
        --sweeps $n_sweep \
        --warmup $n_warmup \
        --beta_min 0.1 \
        --beta_max 6.0 \
        --beta_steps 10 \
        --dims 6 8 8 8 \
        --loops 2 4 \
        --loop_axes 1 2 \
        --polyakov \
        --charge_plaq \
        --charge_clov

# 8x16^3 dataset (test)
mkdir "/media/data/dmueller/ym_datasets/D4_8x16p3/"

python generate_dataset.py \
        --paths "/media/data/dmueller/ym_datasets/D4_8x16p3/test.hdf5" \
        --nums $n_testval \
        --sweeps $n_sweep \
        --warmup $n_warmup \
        --beta_min 0.1 \
        --beta_max 6.0 \
        --beta_steps 10 \
        --dims 8 16 16 16 \
        --loops 2 4 \
        --loop_axes 1 2 \
        --polyakov \
        --charge_plaq \
        --charge_clov

# 12x32^3 dataset (test)
mkdir "/media/data/dmueller/ym_datasets/D4_12x32p3/"

python generate_dataset.py \
        --paths "/media/data/dmueller/ym_datasets/D4_12x32p3/test.hdf5" \
        --nums $n_testval \
        --sweeps $n_sweep \
        --warmup $n_warmup \
        --beta_min 0.1 \
        --beta_max 6.0 \
        --beta_steps 10 \
        --dims 12 32 32 32 \
        --loops 2 4 \
        --loop_axes 1 2 \
        --polyakov \
        --charge_plaq \
        --charge_clov
