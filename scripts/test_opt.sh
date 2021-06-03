#!/usr/bin/env bash

set -x

all_args=("$@")

#MODEL=$1
GPU=$1
MODEL=$2
ITER=$3
TEST_MOD=$4
DATASET=$5

#echo $DATASET_rest
echo $all_args
#rest_args=("${all_args[@]:5}")

ITER=${ITER}

echo $MODEL
python train_test.py \
    --flagfile /glusterfs/yufeiy2/transfer/HoloGAN/${MODEL}/flags.txt \
    --checkpoint ${MODEL}/${ITER}.pth \
    --train 0 \
    --batch_size 1 \
    --dataset ${DATASET} \
    --test_mod ${TEST_MOD}_opt \
    --lap_loss 100  --lap_norm_loss .1 --cyc_mask_loss 10 --cyc_perc_loss 0 \
    --gpu $GPU


#    --batch_size 8 \

