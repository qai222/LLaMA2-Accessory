#!/bin/bash

# path to original LLaMA2
pretrained_path=$PWD/pretrained/Llama-2-7b
tokenizer_path=$PWD/pretrained/Llama-2-7b/tokenizer.model

pretrained_type=meta_ori


llama_config=$PWD/llamaAdapter.json
data_config=$PWD/ord_data_config_exp1.yaml

data_parallel=sdp
model_parallel=1

exp_name=expt_$dataset_name_$(date -d "today" +"%Y%m%d%H%M")

echo "exp name: $exp_name"
mkdir -p output/"$exp_name"

cp finetune_adapter.sh output/"$exp_name"/

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=1112 --nproc_per_node=3 finetune_adapter.py \
--output_dir output/"$exp_name" --epochs 15 --warmup_epochs 1 \
--batch_size 1 --accum_iter 2 --num_workers 4 \
--max_words 2048 \
--lr 0.00005 --min_lr 0.000005 --clip_grad 2 --weight_decay 0.02 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" \
--llama_type llama_adapter --llama_config $llama_config --tokenizer_path "$tokenizer_path" \
--no_visual \
--pretrained_path "$pretrained_path" --pretrained_type="$pretrained_type" \
--data_config $data_config \
2>&1 | tee -a output/"$exp_name"/output.log

echo "exp name: $exp_name"

