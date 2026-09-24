import importlib
import json
import logging
import os
import shutil
import sys
import time
from typing import Any, Dict
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import functools
import numpy as np
import random
from torch.optim import AdamW
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model.lam.model import DINO_LAM_SOFT
from dataset.lerobot_dataset_pretrain_mp import create_lerobot_dataset, LargeScaleWeightedRandomSampler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from dataset.utils import concat_pad_data_collator
from torch.utils.data import Dataset


accelerator = Accelerator()


def flatten_cfg(cfg: DictConfig) -> dict:
    """Flatten a Hydra DictConfig into the flat dict the training code expects.

    The yaml `model` / `data` / `trainer` sections are merged into a single
    level, alongside the top-level orchestration keys (task_name, save_dir, ...).
    Override any key on the command line with Hydra's key=value syntax, e.g.
    `trainer.max_steps=100000` or `save_dir=checkpoints/foo`. resolve=True expands
    any ${...} interpolations; the result is a plain dict (json-serializable).
    """
    raw = OmegaConf.to_container(cfg, resolve=True)
    config = {}
    for section in ("model", "data", "trainer"):
        config.update(raw.get(section) or {})
    for key, value in raw.items():
        if key not in ("model", "data", "trainer"):
            config[key] = value
    return config


def build_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    lr = config.get("lr", 1e-4)
    weight_decay = config.get("weight_decay", 1e-2)
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)



def save_checkpoint(save_dir, step, model_engine, loss, accelerator, config=None):
    tag = f"step_{step}"
    checkpoint_dir = os.path.join(save_dir, tag)

    if accelerator.is_main_process and os.path.exists(checkpoint_dir):
        logging.warning(f"Checkpoint directory {checkpoint_dir} exists. Removing before overwrite.")
        shutil.rmtree(checkpoint_dir)

    accelerator.wait_for_everyone()

    client_state = {
       "step": step,
       "best_loss": loss if isinstance(loss, float) else loss.item(),
       "config": config,
    } if accelerator.is_main_process else {}

    model_engine.save_checkpoint(save_dir, tag=tag, client_state=client_state)

    if accelerator.is_main_process:
        if config is not None:
            config_path = os.path.join(checkpoint_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

        checkpoint_meta_path = os.path.join(checkpoint_dir, "checkpoint.json")
        checkpoint_meta = {
            "type": "ds_model",
            "version": 0.0,
            "checkpoints": "mp_rank_00_model_states.pt"
        }
        with open(checkpoint_meta_path, "w") as f:
            json.dump(checkpoint_meta, f, indent=2)
        logging.info(f"[Rank {accelerator.process_index}] Saved checkpoint to {checkpoint_dir}")


def inspect_named_submodules(module_dict: dict, verbose: bool = True):

    total_all, trainable_all = 0, 0
    logging.info("\n Parameter Inspection by Module:")
    logging.info("=" * 70)
    for module_name, module in module_dict.items():
        total, trainable = 0, 0
        logging.info(f"\n Module: {module_name}")
        logging.info("-" * 70)
        for name, param in module.named_parameters():
            num_params = param.numel()
            total += num_params
            if param.requires_grad:
                trainable += num_params
                if verbose:
                    logging.info(f"Trainable {name:55s} | shape: {str(tuple(param.shape)):20s} | {num_params/1e6:6.2f}M")
            elif verbose:
                logging.info(f"Frozen {name:55s} | shape: {str(tuple(param.shape)):20s} | {num_params/1e6:6.2f}M")
        logging.info("-" * 70)
        logging.info(f"Total     : {total / 1e6:.2f}M")
        logging.info(f"Trainable : {trainable / 1e6:.2f}M")
        logging.info(f"Frozen    : {(total - trainable) / 1e6:.2f}M")
        total_all += total
        trainable_all += trainable
    logging.info("=" * 70)
    logging.info(f"ALL TOTAL     : {total_all / 1e6:.2f}M")
    logging.info(f"ALL TRAINABLE : {trainable_all / 1e6:.2f}M")
    logging.info(f"ALL FROZEN    : {(total_all - trainable_all) / 1e6:.2f}M")
    logging.info("=" * 70)

def main(config: dict) -> None:
    # config is a flat dict (yaml sections + CLI key=value overrides) from flatten_cfg().

    set_seed(42 + accelerator.process_index)

    if accelerator.is_main_process:
        wandb.init(project=os.environ.get('WANDB_PROJECT', 'RotVLA'),
            name=config["task_name"], reinit=True, mode="online", dir='wandb/' + config["task_name"])

    stage = config.get("stage", "stage1-latent-pretrain")

    vla_dataset, training_ds_weights = create_lerobot_dataset(
        config,
        tokenizer=None,
        stage=stage,
        image_horizon=config.get("image_horizon", 3),
        is_log=accelerator.is_main_process,
    )

    collator = functools.partial(concat_pad_data_collator, stage=stage)

    sampler = LargeScaleWeightedRandomSampler(training_ds_weights, len(vla_dataset),num_samples_part=len(vla_dataset)) # 1500*8*32
    dataloader = DataLoader(
        vla_dataset,
        batch_size=config.get("batch_size"),
        sampler=sampler,
        collate_fn=collator,
        num_workers=config.get("num_workers", 8),
        shuffle=False,
        persistent_workers=False,
    )

    model = DINO_LAM_SOFT(**config)

    if accelerator.is_main_process:
        inspect_named_submodules({
                "latent_action_model": model.lam,
            })
    if accelerator.is_main_process:
        print('config', config)
        print('total length of dataset',len(vla_dataset))
        print('model.parameters()',model.parameters())
    optimizer = build_optimizer(model, config)


    #max_epochs = config.get("max_epochs", 1)

    ckpt_every = config.get("ckpt_interval", 1)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, dataloader)
    if accelerator.is_main_process:
        print("Initialized with Accelerate")

    epoch_len = len(train_loader)

    max_steps = config["max_steps"]
    model.train()
    num_batches = 0
    while num_batches < max_steps:

        start_time = time.time()
        for _, batch in enumerate(train_loader):

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss = model.training_step(
                    batch, num_batches,
                    start_time=start_time, epoch_len=epoch_len
                )
            num_batches += 1

            start_time = time.time()

            accelerator.backward(loss)
            optimizer.step()

            if ckpt_every > 0 and num_batches % ckpt_every == 0 and num_batches > 0:
                save_checkpoint(
                    save_dir=config["save_dir"],
                    step=num_batches,
                    model_engine=model,
                    loss=loss,
                    accelerator=accelerator,
                    config=config,
                )

            if num_batches > max_steps:
                break
   

@hydra.main(config_path="../config", config_name="lam-stage-2-lerobot-latent", version_base="1.3")
def entry(cfg: DictConfig):
    config = flatten_cfg(cfg)
    main(config)


if __name__ == "__main__":
    entry()
