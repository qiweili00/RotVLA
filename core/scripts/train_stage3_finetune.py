import sys
import os
import math
from torch import amp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from model.rotvla import RotVLA
import logging
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator, DistributedType
import json
import shutil
from torch.optim import AdamW
import functools
from collections import deque

from dataset.utils import concat_pad_data_collator
import warnings
import numpy as np
import random
import torch.backends.cudnn as cudnn
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
accelerator = Accelerator()
from torchvision.utils import make_grid
from PIL import Image
def get_with_warning(config: dict, key: str, default):
    if key in config:
        return config[key]
    else:
        warnings.warn(f"'{key}' not found in config, using default: {default!r}")
        return default


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



def get_lr_lambda(warmup_steps, total_steps, resume_step=0):
    def lr_lambda(current_step):
        current_step += resume_step  
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda
    
def setup_logging(log_dir: str) -> str:
    from datetime import datetime
    import logging, os

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_log_{timestamp}.log")
    if accelerator is None or accelerator.is_main_process:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        logging.info(f"Logging to: {log_path}")
    return log_path



def init_wandb(config: dict, accelerator: Accelerator):

    if accelerator is None or accelerator.is_main_process:
        wandb.init(
            project=os.environ.get('WANDB_PROJECT', 'RotVLA'),
            name=config.get("run_name", "default_run"),
            mode="online",
            dir='wandb/'+config["run_name"],
        )



def prepare_dataloader(dataset, config: dict,collate_fn,shuffle=True) -> DataLoader:
    batch_size = get_with_warning(config, "batch_size", 8)
    num_workers = get_with_warning(config, "num_workers", 8)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        collate_fn=collate_fn
    )
    if accelerator is None or accelerator.is_main_process:
        logging.info(f"Initialized dataloader with batch size {batch_size}")
    return dataloader


def check_numerical_stability(step: int, **named_tensors) -> bool:
    for name, tensor in named_tensors.items():
        if not torch.isfinite(tensor).all():
            logging.info(f"[Step {step}] Non-finite detected in {name}")
            return False
    return True


def log_training_step(step, loss, total_norm, clipped_norm, scheduler, dataloader, accelerator, loss_extra=None):
    current_epoch = step / len(dataloader)
    if accelerator is None or accelerator.is_main_process:

        logging.info(f"Estimated Epoch: {current_epoch:.2f}")
        if 'latent_action_loss' in loss_extra.keys():
            logging.info(f"[Step {step}] Loss: {loss.item():.4f}, Latent Loss: {loss_extra['latent_action_loss'].item():.4f}")
        else :
            logging.info(f"[Step {step}] Loss: {loss.item():.4f}")

        log_dict = {
            "step": step,
            "loss": loss.item(),
            "current_epoch": current_epoch,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        if loss_extra is not None:
            for key, value in loss_extra.items():
                log_dict[key] = value.item()
            
        wandb.log(log_dict,step=step)

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

def load_checkpoint_with_deepspeed(model_engine, load_dir, accelerator, tag="step_best", load_optimizer_states=True):

    try:
        load_path, client_state = model_engine.load_checkpoint(
            load_dir,
            tag=tag,
            load_module_strict=True,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_optimizer_states
        )
        if accelerator.is_main_process:
            logging.info(f"Loaded DeepSpeed checkpoint from {load_dir}/{tag} (including optimizer states)")
        return client_state.get("step", 0), client_state
        
    except Exception as e:
        if accelerator.is_main_process:
            logging.warning(f"World size mismatch detected: {str(e)}")
            logging.warning("Attempting to load only model weights (skipping optimizer states)...")
        try:
            load_path, client_state = model_engine.load_checkpoint(
                load_dir,
                tag=tag,
                load_module_strict=True,
                load_optimizer_states=False,
                load_lr_scheduler_states=False
            )
            if accelerator.is_main_process:
                logging.info(f"Loaded DeepSpeed checkpoint from {load_dir}/{tag} (model weights only)")
            return client_state.get("step", 0), client_state
            
        except Exception as e2:
            if accelerator.is_main_process:
                logging.warning(f"World size and model mismatch detected: {str(e2)}")
                logging.warning("Attempting to load only model weights (skipping optimizer states and some weight)...")
            try :
                load_path, client_state = model_engine.load_checkpoint(
                    load_dir,
                    tag=tag,
                    load_module_strict=False,
                    load_optimizer_states=False,
                    load_lr_scheduler_states=False
                )
                if accelerator.is_main_process:
                    logging.info(f"Loaded DeepSpeed checkpoint from {load_dir}/{tag} (model weights only strict false)")
                return client_state.get("step", 0), client_state
            except Exception as e3:
                if accelerator.is_main_process:
                    logging.error(f"Failed to load checkpoint even without optimizer states and strict false: {str(e3)}")
                raise RuntimeError(f"Failed to load DeepSpeed checkpoint from {load_dir} with tag {tag}: {str(e3)}")

    

def get_and_clip_grad_norm(accelerator, model, loss, max_norm: float = 1.0):

    if hasattr(accelerator, "get_global_grad_norm") and hasattr(accelerator, "clip_grad_norm_"):
       
        total_norm = accelerator.get_global_grad_norm()
        accelerator.clip_grad_norm_(model.parameters(), max_norm)
        clipped_norm = accelerator.get_global_grad_norm()
    else:
 
        grad_norms = [p.grad.norm(2) for p in model.parameters() if p.grad is not None]
        if len(grad_norms) == 0:
            total_norm = torch.tensor(0.0, device=loss.device)
        else:
            total_norm = torch.norm(torch.stack(grad_norms), 2)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        clipped_grad_norms = [p.grad.norm(2) for p in model.parameters() if p.grad is not None]
        if len(clipped_grad_norms) == 0:
            clipped_norm = torch.tensor(0.0, device=loss.device)
        else:
            clipped_norm = torch.norm(torch.stack(clipped_grad_norms), 2)

    return total_norm, clipped_norm

def build_param_groups(model, wd,lr, lr_scale=1.0):
    decay, no_decay,decay_scale, no_decay_scale = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        is_bias = n.endswith("bias") or ".bias" in n
        is_norm = (p.dim() == 1) or ("norm" in n.lower())
        if n.startswith("action_head"):
            #logging.info(f"action head: {n}")
            (no_decay if is_bias or is_norm else decay).append(p)
            
        else:
            (no_decay_scale if is_bias or is_norm else decay_scale).append(p)
        
    return [{"params": decay,"lr": lr , "weight_decay": wd},
            {"params": no_decay, "lr": lr , "weight_decay": 0.0},
            {"params": decay_scale,"lr": lr / lr_scale, "weight_decay": wd},
            {"params": no_decay_scale, "lr": lr / lr_scale, "weight_decay": 0.0}]

def load_checkpoint_before_prepare(model, ckpt_dir):

    rank0_ckpt = os.path.join(ckpt_dir, "mp_rank_00_model_states.pt")

    if accelerator.is_main_process:
        logging.info(f"Loading checkpoint: {rank0_ckpt}")


    ckpt = torch.load(rank0_ckpt, map_location="cpu",weights_only=False)
    ckpt_sd = ckpt["module"]   # DeepSpeed 默认字段名

    model_sd = model.state_dict()
    filtered = {}

    for k, v in ckpt_sd.items():

        if "latent_action_model" in k:
            if accelerator.is_main_process:
                logging.info(f"Skip param: {k}")
            continue

        if k in model_sd and v.shape != model_sd[k].shape:
            if accelerator.is_main_process:
                logging.info(
                    f"Skip mismatched param: {k}, "
                    f"ckpt={v.shape}, model={model_sd[k].shape}"
                )
            continue

        if k in model_sd:
            filtered[k] = v
        else:
            if accelerator.is_main_process:
                logging.info(f"Drop unknown key: {k}")

    
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if accelerator.is_main_process:
        logging.info('latent action are loaded when creating model')
        filtered_missing = [k for k in missing if "latent_action_model" not in k]
        logging.info(f"Missing keys: {filtered_missing}")
   
        logging.info(f"Unexpected keys: {unexpected}")


    return model

from typing import Any, Dict, Sequence
from dataset.lerobot_dataset_pretrain_mp import create_lerobot_dataset


def flatten_cfg(cfg: DictConfig) -> dict:
    """Flatten a Hydra DictConfig into the flat dict the training code expects.

    The yaml `model` / `data` / `trainer` sections are merged into a single
    level, alongside the top-level orchestration keys (run_name, save_dir, ...).
    Override any key on the command line with Hydra's key=value syntax, e.g.
    `trainer.lr=2e-4` or `save_dir=checkpoints/foo`. resolve=True expands any
    ${...} interpolations; the result is a plain dict (json-serializable).
    """
    raw = OmegaConf.to_container(cfg, resolve=True)
    config = {}
    for section in ("model", "data", "trainer"):
        config.update(raw.get(section) or {})
    for key, value in raw.items():
        if key not in ("model", "data", "trainer"):
            config[key] = value
    return config


def train(config):


    # === Set logging ===
    save_dir = get_with_warning(config, "save_dir", "checkpoints")
    log_path = setup_logging(save_dir)
    
    # === WandB ===
    init_wandb(config, accelerator)
    logging.info(f"[DEBUG] process_index={accelerator.process_index}, num_processes={accelerator.num_processes}, is_main={accelerator.is_main_process}")
    # === Debug mode ===
    if get_with_warning(config, "debug", False):
        torch.autograd.set_detect_anomaly(True)

    # config is a flat dict (yaml sections + CLI key=value overrides) from flatten_cfg()

    # === Resume training from checkpoint ===
    resume_path = get_with_warning(config, "resume_path", None)
    resume_from_stage2 = get_with_warning(config, "resume_from_stage2", False)


    latent_supervision = get_with_warning(config, "latent_supervision", "z_m")
    assert latent_supervision in ("z_m", "z_q"), \
        f"latent_supervision must be 'z_m' or 'z_q', got {latent_supervision!r}"
    if accelerator.is_main_process:
        logging.info(f"Latent supervision target: {latent_supervision}")

    if config["seed"] > 0:
        set_seed(config["seed"]+accelerator.process_index)
    # === Model ===
    model = RotVLA(config)
    if resume_path is not None and resume_from_stage2:
        
        model = load_checkpoint_before_prepare(model,resume_path)
    # === Dataset ===
    dataset, training_ds_weights = create_lerobot_dataset(
        config,
        tokenizer=model.embedder.tokenizer,
        stage=config["stage"],
        horizon=config["horizon"],
        image_horizon=config["image_horizon"],
        is_log=accelerator.is_main_process,
        shuffle=True,
    )


    collator = functools.partial(concat_pad_data_collator, pad_id=model.embedder.tokenizer.pad_token_id,stage=config["stage"])
    # === DataLoader ===
    dataloader = prepare_dataloader(dataset, config,collate_fn=collator,shuffle=False)

    
    model.train()
    model.set_finetune_flags()

    lr = config["lr"]
    wd = config["weight_decay"]
    lr_scale = config["lr_scale"]
    optimizer = AdamW(build_param_groups(model, wd, lr, lr_scale=lr_scale)) #, lr=lr)
    if accelerator.is_main_process:
        logging.info(f"Optimizer=AdamW, lr={lr}, weight_decay={wd}")

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model_engine = model  
    
    if accelerator.is_main_process:
        logging.info("Initialized with Accelerate")
        #logging.info(f"config: {config}")
    
    
    # === Warmup + Cosine Scheduler ===
    max_steps = config["max_steps"]
    warmup_steps = config["warmup_steps"]
    
    # === loss function ===
    #loss_fn = nn.MSELoss() 

    # === Checkpoint and save path setup ===
    os.makedirs(save_dir, exist_ok=True)
    
    # === Logging and interval settings ===
    log_interval = config["log_interval"]
    ckpt_interval = config["ckpt_interval"]
    max_norm = config["grad_clip_norm"]

    


    # When True, resume the model/optimizer weights but reset the step counter to 0,
    # so the LR scheduler re-runs warmup and training runs a full max_steps.
    begin_with_step0 = get_with_warning(config, "begin_with_step0", False)

    if resume_path is not None and not resume_from_stage2:
        resume_path = resume_path.rstrip("/")
        resume_dir, resume_tag = os.path.split(resume_path)

        step, client_state = load_checkpoint_with_deepspeed(
            model_engine,
            load_dir=resume_dir,
            accelerator=accelerator,
            tag=resume_tag,
            load_optimizer_states=True,
        )
        if begin_with_step0:
            if accelerator.is_main_process:
                logging.info(f"Resumed weights from {resume_dir}/{resume_tag} (loaded step {step}), "
                             f"resetting step to 0 to restart warmup")
            step = 0
        elif accelerator.is_main_process:
            logging.info(f"Resuming from {resume_dir}/{resume_tag}, step {step}")
    else:
        step = 0
        if accelerator.is_main_process:
            logging.info("Starting fresh training")


    scheduler = LambdaLR(optimizer, get_lr_lambda(warmup_steps, max_steps, resume_step=step))


    if accelerator.is_main_process:
        inspect_named_submodules({
            "vision_model": model.embedder.model.vision_model,
            "language_model": model.embedder.model.language_model,
            "action_head": model.action_head,
            #"latent action":model.latent_action_model,
        })



    while step < max_steps:
        import time
        #start_time = time.time()
        for batch in tqdm(dataloader, desc="Training", disable=not accelerator.is_main_process):
            if step >= max_steps:
                break

            prompts = batch["prompts"]
            states = batch["states"].to(dtype=torch.bfloat16)
            actions_gt = batch["actions"].to(dtype=torch.bfloat16)
            action_mask = batch["action_mask"]
            videos = batch["videos"]
            B = len(prompts)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                

                model.latent_action_model.eval()
                with torch.no_grad():
                    latent_action_dict = model.latent_action_model.get_latent_actions(videos)

                if latent_supervision == "z_q":
                    actions_gt_latent = latent_action_dict['z_q'].squeeze(1).detach()
                else:
                    actions_gt_latent = latent_action_dict['z_m'].detach()
                # actions_gt_latent = latent_action_dict['z_m'].detach()
                # actions_gt_latent = latent_action_dict['z_q'].squeeze(1).detach()
                #print(actions_gt_latent.shape)
                vlm_outputs = model.common_process(pixel_values=batch["pixel_values"],
                                    input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    position_ids=batch["position_ids"],
                                    image_flags=batch["image_flags"],
                                    output_hidden_states=True,)

                fused_tokens = vlm_outputs.hidden_states[-1]



            target_latent_velocity = actions_gt_latent
            pred_velocity, pred_latent_velocity = model(fused_tokens, state=states, actions_gt=actions_gt,action_gt_latent=actions_gt_latent, action_mask=action_mask)

            target_velocity = actions_gt

            

            action_mask = action_mask.view(action_mask.shape[0], -1).to(dtype=pred_velocity.dtype)
            pred_velocity_mask = pred_velocity * action_mask

            target_velocity = model.action_space.preprocess_gt(target_velocity)
            loss = model.action_space.compute_loss(pred_velocity_mask.reshape(B,model.num_actions, -1), target_velocity)
            loss_extra = {
                "position_loss": loss["position_loss"],
                "rotate6D_loss": loss["rotate6D_loss"],
                "gripper_loss": loss["gripper_loss"],
            }
            loss_action = loss["action_loss"].clone()

            loss_latent = model.action_space.compute_latent_action_loss(pred_latent_velocity.reshape(B,model.num_latent_actions, -1), target_latent_velocity)
            loss = loss["action_loss"] + loss_latent["action_loss"]
            loss_extra["latent_action_loss"] = loss_latent["action_loss"]

            
            if action_mask.sum() == 0:
                raise ValueError(f"[Step {step}] action_mask.sum() is 0! All actions are masked. "
                            f"This indicates a problem with the data or mask generation. "
                            f"action_mask shape: {action_mask.shape}, "
                            f"action_mask: {action_mask}")

            # === NaN/Inf check ===
            if not check_numerical_stability(
                step,
                states=states,
                actions_gt=actions_gt,
                fused_tokens=fused_tokens,
                pred_velocity=pred_velocity,
                loss=loss
            ):
                continue

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)

            total_norm, clipped_norm = get_and_clip_grad_norm(accelerator, model, loss, max_norm)
            optimizer.step()

            scheduler.step()


            if step % log_interval == 0:
                log_training_step(step, loss_action, total_norm, clipped_norm, scheduler, dataloader, accelerator, loss_extra=loss_extra)
   


            step += 1
            


            if step % 500 == 0 :
                model.eval()
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        _, pred_velocity = model(fused_tokens, state=states, actions_gt=None, action_mask=None)

                        pred_velocity = pred_velocity.reshape_as(latent_action_dict['z'])
                        zq_identity = model.latent_action_model.identity.repeat(pred_velocity.shape[0],1,1,1).detach()
                      
                        latent_action_dict_decode = model.latent_action_model.decode_latent(videos, latent_action_dict['z_q'],pred_velocity,zq_identity)
                        
                image = torch.cat([videos[:4,0],videos[:4,1],latent_action_dict_decode["video_recon"][:4],latent_action_dict_decode["video_recon2"][:4],latent_action_dict_decode["video_identity"][:4]], dim=0)
                image = torch.clamp(image, min=0, max=1)
                image = make_grid(image, nrow=4, padding=0, pad_value=1.0)
                image = image.permute(1, 2, 0).mul_(255).cpu().numpy()
                image = Image.fromarray(image.astype(np.uint8))
                if accelerator is None or accelerator.is_main_process:
                    wandb.log({"recon_images": wandb.Image(image,caption='1.'+batch['prompts'][0]+'\n'+' 2.'+batch['prompts'][1]+'\n'+' 3.'+batch['prompts'][2]+'\n'+' 4.'+batch['prompts'][3])}, step=step)

                model.train()
                model.latent_action_model.eval()



            # === Save periodic checkpoint ===
            if step % ckpt_interval == 0 and step > 0:
                save_checkpoint(save_dir, step=step, model_engine=model_engine, loss=loss, accelerator=accelerator, config=config)




@hydra.main(config_path="../config", config_name="lam-stage-2-lerobot-224-finetune", version_base="1.3")
def main(cfg: DictConfig):
    config = flatten_cfg(cfg)
    try:
        train(config)
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            logging.info("KeyboardInterrupt received. Cleaning up...")
        sys.exit(0)


if __name__ == "__main__":
    main()

