#!/bin/bash
export PYTHONWARNINGS="ignore"
n_models=5
max_epochs=50
lr=1e-2
weight_decay=0.0
batch_size=10

# LGE-CNN

python train.py \
        --name "D4_QP" \
        --logdir "../logs/wilson/" \
        --num_models $n_models \
        --max_epochs $max_epochs \
        --lr $lr \
        --weight_decay $weight_decay \
        --batch_size $batch_size \
        --dims 4 8 8 8  \
        --conv_channels 4 \
        --conv_kernel_size 2 \
	--symmetric \
        --conv_dilation 0 \
        --out_mode "QP" \
        --train_path "/media/data/dmueller/ym_datasets/D4_4x8p3/train.hdf5" \
        --val_path "/media/data/dmueller/ym_datasets/D4_4x8p3/val.hdf5" \
        --test_path "/media/data/dmueller/ym_datasets/D4_4x8p3/test.hdf5" \
        --gpus 1 \

