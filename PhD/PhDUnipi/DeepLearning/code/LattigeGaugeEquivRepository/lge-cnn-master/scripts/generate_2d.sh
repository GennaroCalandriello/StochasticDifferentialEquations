export PYTHONWARNINGS="ignore"

# 8x8 dataset (training, validation and test)
mkdir "/media/data/dmueller/ym_datasets/D2_8/"

python generate_dataset.py \
        --paths "/media/data/dmueller/ym_datasets/D2_8/train.hdf5" \
        "/media/data/dmueller/ym_datasets/D2_8/val.hdf5" \
        "/media/data/dmueller/ym_datasets/D2_8/test.hdf5" \
        --nums 1000 100 100 \
        --sweeps 100 \
        --warmup 2000 \
        --beta_min 0.1 \
        --beta_max 6.0 \
        --beta_steps 10 \
        --dims 8 8 \
        --loops 2 4 \
        --loop_axes 0 1 \
        --polyakov


# 16x16 dataset (test)
mkdir "/media/data/dmueller/ym_datasets/D2_16/"

python generate_dataset.py \
        --paths "/media/data/dmueller/ym_datasets/D2_16/test.hdf5" \
        --nums 100 \
        --sweeps 100 \
        --warmup 2000 \
        --beta_min 0.1 \
        --beta_max 6.0 \
        --beta_steps 10 \
        --dims 16 16 \
        --loops 2 4 \
        --loop_axes 0 1 \
        --polyakov


# 32x32 dataset (test)
mkdir "/media/data/dmueller/ym_datasets/D2_32/"

python generate_dataset.py \
        --paths "/media/data/dmueller/ym_datasets/D2_32/test.hdf5" \
        --nums 100 \
        --sweeps 100 \
        --warmup 2000 \
        --beta_min 0.1 \
        --beta_max 6.0 \
        --beta_steps 10 \
        --dims 32 32 \
        --loops 2 4 \
        --loop_axes 0 1 \
        --polyakov

# 64x64 dataset (test)
mkdir "/media/data/dmueller/ym_datasets/D2_64/"

python generate_dataset.py \
        --paths "/media/data/dmueller/ym_datasets/D2_64/test.hdf5" \
        --nums 100 \
        --sweeps 100 \
        --warmup 2000 \
        --beta_min 0.1 \
        --beta_max 6.0 \
        --beta_steps 10 \
        --dims 64 64 \
        --loops 2 4 \
        --loop_axes 0 1 \
        --polyakov