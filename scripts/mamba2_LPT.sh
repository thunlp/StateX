#!/bin/bash

export TOKENIZERS_PARALLELISM="false"

echo "STARTING TRAIN"

args="$@"

for arg in "$@"; do
    eval "$arg"
done

# Arguments with default values
n_gpus=${n_gpus:-8}
exp_group=${exp_group:-"mamba2-1.3B-LPT"}
proj_name=${proj_name:-"LPT"}
train_config=${train_config:-"drop/10b_bsz-512k_64k_cos_3e-4_slimpj"}
comment=${comment:-""}
# Some arguments need to be constructed
run_name="${exp_group}_${model_config}_${train_config}_${comment}"

ckpt_path="huggingface/mamba2-1.3b-hf"

model_name="mamba2"
model_config="${ckpt_path}/config.json"
train_config="./configs/training/${train_config}.json"
tok_path="tokenizer/mamba-tokenizer"

steps=""
init_from="${ckpt_path}${steps}/model.safetensors"

# Build command
cmd="accelerate launch --num_processes=${n_gpus} --main_process_port=29502 train.py"

cmd+=" --run_name=${run_name}"
cmd+=" --proj_name=${proj_name}"
cmd+=" --init_from=${init_from}"
cmd+=" --tok_path=${tok_path}"

# Add command-line arguments to the cmd string
for arg in "$@"; do
    cmd+=" --$arg"
done

cmd+=" --model_name=${model_name}"
cmd+=" --model_config=${model_config}"
cmd+=" --train_config=${train_config}"
cmd+=" --grad_ckpt=1"


echo "======== Final command ========"
echo "$cmd" | tr ' ' '\n'
echo "==============================="

export WANDB_MODE=offline

$cmd
