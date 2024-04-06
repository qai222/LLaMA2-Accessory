#!/bin/bash

pretrained_path=$PWD/output/expt_202403020036/epoch14
output_path=$PWD/infer/expt_202403020036/epoch14
test_path=$PWD/USPTO-n100k-t2048_exp1/test.json
mkdir -p $output_path

# batch mode
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master-port=1222 infer_batch.py \
--pretrained_path $pretrained_path \
--test_json_path $test_path \
--infer_batch_size 8 \
--output_path $output_path

## app mode
#NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=1 --master-port=1222 infer_app.py \
#--pretrained_path $pretrained_path \
#--max_seq_len 2048
