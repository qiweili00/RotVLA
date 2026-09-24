#!/bin/bash
# Stage 3: finetune the pretrained RotVLA on RoboTwin, in two phases.
#   phase 1 - frozen VLM, train the action head only, from the Stage 2 checkpoint
#   phase 2 - unfreeze the VLM and train jointly, resuming from phase 1
#
# Config: config/lam-stage-2-lerobot-224-finetune.yaml (shared by all Stage 3 tasks)
# Set LAM_PATH / VLA_PRETRAIN to your Stage 1 / Stage 2 checkpoints before running.
#
# Override any config key inline with Hydra's key=value syntax:
#   - orchestration:  run_name=... save_dir=... lam_path=... resume_path=...
#   - task knobs:     trainer.action_space=... trainer.data_mix=...
#   - training:       trainer.lr=... trainer.max_steps=... trainer.horizon=...
# Switch config file with --config-name=<name> (no .yaml suffix).
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-8}
LAM_PATH=${LAM_PATH:-checkpoints/LAM-Pretrain/stage1_ckpt.pt}
VLA_PRETRAIN=${VLA_PRETRAIN:-checkpoints/VLA-Pretrain}



PHASE1=robotwin-rotvla-phase1
PHASE2=robotwin-rotvla-phase2

# --- phase 1: frozen VLM ---
accelerate launch --num_processes "$NUM_GPUS" --num_machines 1 \
    --config_file config/accelerate_config/default_config.yaml \
    scripts/train_stage3_finetune.py \
    run_name="$PHASE1" \
    save_dir=checkpoints/"$PHASE1" \
    lam_path="$LAM_PATH" \
    trainer.batch_size=32 \
    trainer.max_steps=5000 \
    trainer.ckpt_interval=5000 \
    trainer.finetune_vlm=false \
    trainer.action_space=ee6d_robotwin \
    trainer.use_augmentation=true \
    trainer.data_mix=robotwin-all-100 \
    resume_from_stage2=true \
    resume_path="$VLA_PRETRAIN"

# --- phase 2: finetune VLM + action head ---
accelerate launch --num_processes "$NUM_GPUS" --num_machines 1 \
    --config_file config/accelerate_config/default_config.yaml \
    scripts/train_stage3_finetune.py \
    run_name="$PHASE2" \
    save_dir=checkpoints/"$PHASE2" \
    lam_path="$LAM_PATH" \
    trainer.batch_size=32 \
    trainer.max_steps=120000 \
    trainer.finetune_vlm=true \
    trainer.action_space=ee6d_robotwin \
    trainer.use_augmentation=true \
    trainer.data_mix=robotwin-all-100 \
    trainer.begin_with_step0=true \
    resume_path=checkpoints/"$PHASE1"/step_5000
