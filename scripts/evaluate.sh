#!/bin/bash

# Evaluate DynamicsNet (GRIP) inference output against the PRISM test set.
# Pairs with scripts/dynamics_test.sh — reads results_{success,failed}/ that
# scripts/dynamics_test.sh writes under output/dynamics_net/.

python evaluation/eval_dynamics.py \
    --results-dir output/dynamics_net \
    --gt-dir data/preprocessed/dynamics_net/test \
    --out-dir output/evaluation \
    --seq-len 500
