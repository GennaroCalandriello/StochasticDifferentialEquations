#!/bin/bash
export PYTHONWARNINGS="ignore"
n_models=10
max_epochs=100
lr=1e-3
weight_decay=0.0
batch_size=100

# Small model

python train.py \
        --name "D2_W2_small" \
        --logdir "../logs/wilson/" \
        --num_models $n_models \
        --max_epochs $max_epochs \
        --lr $lr \
        --weight_decay $weight_decay \
        --batch_size $batch_size \
        --dims 8 8 \
        --conv_channels 2 2 \
        --conv_kernel_size 2 2 \
        --conv_dilation 0 0 \
        --out_mode "trW_2" \
        --train_path "/media/data/dmueller/ym_datasets/D2_8/train.hdf5" \
        --val_path "/media/data/dmueller/ym_datasets/D2_8/val.hdf5" \
        --test_path "/media/data/dmueller/ym_datasets/D2_8/test.hdf5" \
        --gpus 1 \
        
# Medium model

python train.py \
        --name "D2_W2_medium" \
        --logdir "../logs/wilson/" \
        --num_models $n_models \
        --max_epochs $max_epochs \
        --lr $lr \
        --weight_decay $weight_decay \
        --batch_size $batch_size \
        --dims 8 8 \
        --conv_channels 4 4 \
        --conv_kernel_size 3 3 \
        --conv_dilation 0 0 \
        --out_mode "trW_2" \
        --train_path "/media/data/dmueller/ym_datasets/D2_8/train.hdf5" \
        --val_path "/media/data/dmueller/ym_datasets/D2_8/val.hdf5" \
        --test_path "/media/data/dmueller/ym_datasets/D2_8/test.hdf5" \
        --gpus 1 \
        

        
# Large model

python train.py \
        --name "D2_W2_large" \
        --logdir "../logs/wilson/" \
        --num_models $n_models \
        --max_epochs $max_epochs \
        --lr $lr \
        --weight_decay $weight_decay \
        --batch_size $batch_size \
        --dims 8 8 \
        --conv_channels 8 8 \
        --conv_kernel_size 4 4 \
        --conv_dilation 0 0 \
        --out_mode "trW_2" \
        --train_path "/media/data/dmueller/ym_datasets/D2_8/train.hdf5" \
        --val_path "/media/data/dmueller/ym_datasets/D2_8/val.hdf5" \
        --test_path "/media/data/dmueller/ym_datasets/D2_8/test.hdf5" \
        --gpus 1 \
        

