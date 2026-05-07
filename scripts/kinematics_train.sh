#!/bin/bash

# KinematicsNet training script
# Runs independent training followed by joint training

echo "=== KinematicsNet training started ==="
echo "Started at: $(LC_ALL=C date)"

# Stop on error
set -e

# 1. Independent training
echo ""
echo "=== 1. Independent training started ==="
python kinematics_net/train.py \
            --use_wandb true \
            --mode independent \
            --dataset PRISM \
            --insole True \
            --n_imus 4 \
            --cuda_device 0

# Check independent training succeeded
if [ $? -eq 0 ]; then
    echo "Independent training completed successfully"
else
    echo "Independent training failed"
    exit 1
fi

# 2. Joint training
echo ""
echo "=== 2. Joint training started ==="
python kinematics_net/train.py \
            --use_wandb true \
            --mode joint \
            --dataset PRISM \
            --insole True \
            --n_imus 4 \
            --cuda_device 0

# Check joint training succeeded
if [ $? -eq 0 ]; then
    echo "Joint training completed successfully"
else
    echo "Joint training failed"
    exit 1
fi

echo ""
echo "=== All training completed successfully ==="
echo "Finished at: $(LC_ALL=C date)"
