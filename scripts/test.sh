#!/usr/bin/env bash

#MODEL=$1
GPU=$1
MODEL=$2
ITER=$3
TEST_MOD=$4
DATA=$5

ITER=${ITER}

echo $MODEL
python train_test.py \
    --flagfile /glusterfs/yufeiy2/transfer/HoloGAN/${MODEL}/flags.txt \
    --checkpoint ${MODEL}/${ITER}.pth \
    --train 0 \
    --test_mod ${TEST_MOD} \
    --dataset ${DATA} \
    --batch_size 1 \
    --gpu $GPU



#    --lap_loss 1000 --PAD_FRAC 0  \
