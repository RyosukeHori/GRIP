#!/bin/bash
# Run KinematicsNet inference on the PRISM train + test splits with the
# default 4 IMUs + insole configuration.

set -e

for mode in train test; do
    CUDA_VISIBLE_DEVICES=0 python kinematics_net/inference.py \
        --dataset PRISM \
        --mode "$mode" \
        --n_imus 4 \
        --insole true \
        --fps 100
done



