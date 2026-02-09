#!/bin/bash
task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
num_gpus=${5:-8}
shift 5 2>/dev/null  # consume the 5 positional args; remaining "$@" forwarded below

DEBUG=False
save_ckpt=True

torchrun --nproc_per_node=${num_gpus} --master_port=29500 imitate_episodes.py \
    --task_name sim-${task_name}-${task_config}-${expert_data_num} \
    --ckpt_dir ./act_ckpt/act-${task_name}/${task_config}-${expert_data_num} \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 50 \
    --hidden_dim 512 \
    --batch_size 8 \
    --dim_feedforward 3200 \
    --num_epochs 1000 \
    --lr 5e-5 \
    --save_freq 100 \
    --state_dim 14 \
    --seed ${seed} \
    --eval_task_name ${task_name} \
    --eval_task_config ${task_config} \
    --eval_episodes 20 \
    --eval_step_lim 150 \
    "$@"
    # Language conditioning examples:
    # bash train_multigpu.sh <task> <config> <num> <seed> <gpus> --lang_cond_type film --instructions_dir ../../data/<task>/<config>/instructions
    # bash train_multigpu.sh <task> <config> <num> <seed> <gpus> --lang_cond_type token --instructions_dir ../../data/<task>/<config>/instructions
