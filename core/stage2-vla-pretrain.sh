#!/bin/bash
# Stage 2: VLA pretraining on large-scale heterogeneous data, in two phases.
#   phase 1 - frozen VLM, train the action head only
#   phase 2 - unfreeze the VLM and train jointly, resuming from phase 1
#
# Config: config/lam-stage-2-lerobot-224.yaml
# Set LAM_PATH to the Stage 1 LAM checkpoint before running.
#
# Override any config key inline with Hydra's key=value syntax:
#   - orchestration:  run_name=... save_dir=... lam_path=... vlm_name=...
#   - training:       trainer.lr=... trainer.max_steps=... trainer.horizon=...
# Switch config file with --config-name=<name> (no .yaml suffix).
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-8}
VLM_NAME=${VLM_NAME:-OpenGVLab/InternVL3_5-1B}
LAM_PATH=${LAM_PATH:-checkpoints/lam-stage-1/step_200000/mp_rank_00_model_states.pt}

PHASE1=rotvla-pretrain-phase1
PHASE2=rotvla-pretrain-phase2

# --- phase 1: frozen VLM ---
accelerate launch --num_processes "$NUM_GPUS" --num_machines 1 \
    --config_file config/accelerate_config/default_config.yaml \
    scripts/train_stage2_vla.py \
    --config-name=lam-stage-2-lerobot-224 \
    run_name="$PHASE1" \
    save_dir=checkpoints/"$PHASE1" \
    vlm_name="$VLM_NAME" \
    lam_path="$LAM_PATH" \
    trainer.max_steps=5000 \
    trainer.ckpt_interval=5000 \
    trainer.finetune_vlm=false

# --- phase 2: finetune VLM + action head ---
accelerate launch --num_processes "$NUM_GPUS" --num_machines 1 \
    --config_file config/accelerate_config/default_config.yaml \
    scripts/train_stage2_vla.py \
    --config-name=lam-stage-2-lerobot-224 \
    run_name="$PHASE2" \
    save_dir=checkpoints/"$PHASE2" \
    vlm_name="$VLM_NAME" \
    lam_path="$LAM_PATH" \
    trainer.max_steps=200000 \
    trainer.ckpt_interval=10000 \
    trainer.finetune_vlm=true \
    trainer.begin_with_step0=true \
    resume_path=checkpoints/"$PHASE1"/step_5000
