#!/bin/bash
#SBATCH --time=5:59:59
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --job-name=alama_dpo
#SBATCH --output=%x.out
#SBATCH --error=%x.err

module load python/3.10 scipy-stack gcc arrow cuda cudnn opencv

module load gcc/9.3.0 cuda/11.7 

source ~/ENV_TORCH/bin/activate

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
export CUDA_VISIBLE_DEVICES=0,1,2,3

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_ds_zero3_cpu_offload_config.yaml --num_processes=1 peft_lora_seq2seq_accelerate_ds_zero3_offload.py