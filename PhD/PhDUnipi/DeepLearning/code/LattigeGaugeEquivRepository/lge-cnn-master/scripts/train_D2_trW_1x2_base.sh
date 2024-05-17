#!/bin/bash
export PYTHONWARNINGS="ignore"
n_models=10
max_epochs=100
lr=3e-2
weight_decay=0.0
batch_size=50

# Baseline model

python train.py \
        --name "D2_W1x2_baseline_gp" \
        --baseline \
        --logdir "../logs/wilson/" \
        --num_models $n_models \
        --max_epochs $max_epochs \
        --lr $lr \
        --weight_decay $weight_decay \
        --batch_size $batch_size \
        --dims 8 8 \
        --conv_channels 16 16 16  \
        --conv_kernel_size 2 2 2  \
        --global_average \
        --activation 'relu' \
        --out_mode "trW_1x2" \
        --train_path "/media/data/dmueller/ym_datasets/D2_8/train.hdf5" \
        --val_path "/media/data/dmueller/ym_datasets/D2_8/val.hdf5" \
        --test_path "/media/data/dmueller/ym_datasets/D2_8/test.hdf5" \
        --gpus -1 \

