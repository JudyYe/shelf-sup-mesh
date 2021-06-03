#!/usr/bin/env bash

set -x

all_args=("$@")

GPU=$1
MODEL=$2
ITER=$3

echo $all_args

ITER=${ITER}

echo $MODEL
python train_test.py \
    --flagfile outputs/${MODEL}/flags.txt \
    --checkpoint ${MODEL}/${ITER}.pth \
    --train 0 \
    --batch_size 1 \
    --test_mod quali_opt \
    --lap_loss 100  --lap_norm_loss .1 --cyc_mask_loss 10 --cyc_perc_loss 0 \
    --gpu $GPU
