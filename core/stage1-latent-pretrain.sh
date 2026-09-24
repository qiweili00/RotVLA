#!/bin/bash
# Stage 1: pretrain the Rotational Latent Action Model (LAM).
#
# Config: config/lam-stage-2-lerobot-latent.yaml
# Set `stage_one_ckpt` there to the UniVLA lam-stage-1 checkpoint used for
# initialization, and `data.data_mix` to your pretraining mixture.
#
# Override any config key inline with Hydra's key=value syntax:
#   - orchestration:  task_name=... save_dir=...
#   - training:       trainer.lr=... trainer.max_steps=... trainer.ckpt_interval=...
#   - data:           data.data_mix=... data.batch_size=...
# Switch config file with --config-name=<name> (no .yaml suffix).
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-8}

accelerate launch --num_processes "$NUM_GPUS" --num_machines 1 \
    --config_file config/accelerate_config/default_config.yaml \
    scripts/train_stage1_lam.py \
    --config-name=lam-stage-2-lerobot-latent \
    task_name=lam-stage-1 \
    save_dir=checkpoints/lam-stage-1
