from os import listdir, makedirs, path
from typing import Callable, Dict, Iterable, Tuple
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb 
from PIL import Image
from einops import rearrange
from torch import Tensor
from torch.optim import AdamW, Optimizer
from accelerate import PartialState
import torch.nn.functional as F
OptimizerCallable = Callable[[Iterable], Optimizer]

from model.lam.modules import compute_entropy_loss,SoftDINOLatentActionModel
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


import time
from torchvision.utils import make_grid
import torch.distributed as dist
from accelerate import Accelerator
import copy

class DINO_LAM_SOFT(nn.Module):
    """
    Accelerate / DeepSpeed compatible version of DINO_LAM_SOFT.
    Uses accelerate.PartialState instead of torch.distributed directly.
    """

    def __init__(
            self,
            image_channels: int = 3,
            lam_model_dim: int = 512,
            lam_latent_dim: int = 32,
            lam_num_latents: int = 8,
            lam_patch_size: int = 16,
            lam_enc_blocks: int = 8,
            lam_dec_blocks: int = 8,
            lam_num_heads: int = 8,
            lam_dropout: float = 0.0,
            log_interval: int = 1000,
            lam_num_codes: int = 4,
            task_name: str = 'lam_openx',
            stage: str = 'stage-1',
            stage_one_ckpt: str = None,
            entropy_loss_ratio: float = 0.01,
            use_identity: int = 0,
            log_wandb: int = 1,
            type: str = 'train',
            univla_ckpt=False,
            **kwargs,
    ) -> None:
        super(DINO_LAM_SOFT, self).__init__()


        self.lam = SoftDINOLatentActionModel(
            in_dim=image_channels,
            model_dim=lam_model_dim,
            latent_dim=lam_latent_dim,
            num_latents=lam_num_latents,
            patch_size=lam_patch_size,
            enc_blocks=lam_enc_blocks,
            num_codes=lam_num_codes,
            dec_blocks=lam_dec_blocks,
            num_heads=lam_num_heads,
            dropout=lam_dropout,
            use_identity=use_identity,
        )

        self.use_identity = use_identity
        self.loaded_step = 0

        self._state = PartialState()

        if stage_one_ckpt is not None:
            print('load from',stage_one_ckpt)
            map_loc = f"cuda:{self._state.local_process_index}"

            ckpt = torch.load(stage_one_ckpt, map_location=map_loc, weights_only=False)
            if stage_one_ckpt.endswith('.pt'):
                self.loaded_step = ckpt["step"]
                if "model" in ckpt.keys():
                    lam_ckpt = ckpt["model"]
                else :
                    lam_ckpt = ckpt["module"]
                    univla_ckpt = False
            else:
                lam_ckpt = ckpt.get('state_dict', ckpt)
            stage1_ckpt = {}


            for key in lam_ckpt.keys():

                #print(key)
                if 'lam' in key:
                    key2 =copy.deepcopy(key)
                    if ('action_up.' in key or 'to_codebook.' in key) and univla_ckpt:
                        continue

                    elif 'action_latent' in key and not univla_ckpt and (key2.replace("action_latent", "action_latent_controllable") in lam_ckpt.keys() or 'action_latent_controllable' in key): #对于已有的ckpt
                        if 'action_latent_controllable' in key:
                            stage1_ckpt[key.replace("lam.", "").replace("action_latent_controllable", "action_latent")] = lam_ckpt[key]
                        else:
                            continue
                    elif univla_ckpt and 'action_latent' in key :
                        continue
                    else:
                        stage1_ckpt[key.replace("lam.", "")] = lam_ckpt[key]


            try:
                self.lam.load_state_dict(stage1_ckpt, strict=True)
                print('strict load success')
            except Exception as e:
                print(e)
                print('strict load failed, try non-strict load')
                self.lam.load_state_dict(stage1_ckpt, strict=False)

        self.entropy_loss_ratio = entropy_loss_ratio

        self.lam_num_latents = lam_num_latents
        self.log_interval = log_interval
        self.task_name = task_name
        self.latent_dim = lam_latent_dim
        self.lam_num_codes = lam_num_codes
        self.log_wandb = log_wandb


    def shared_step(self, batch: Dict, batch_idx, order=None) -> Tuple:
        outputs = self.lam(batch, order)

        entropy_weight = self.entropy_loss_ratio

        def step(recon_ind=0, target_id=0):
            gt_future_frames = outputs["target"][:, [target_id]]

            gt_future_frames = gt_future_frames[:, 0]
            mse_loss = ((gt_future_frames - outputs["recon"][recon_ind]) ** 2).mean()
            entropy_loss = compute_entropy_loss(outputs["emb"][recon_ind])


            unique, counts = torch.unique(outputs["indices"][recon_ind], return_counts=True)
            index_counts = torch.zeros(self.lam_num_latents, dtype=torch.long).type_as(counts)
            index_counts[unique] = counts
            code_usage = (index_counts != 0).float().mean()

            loss_logs = {
                "mse_loss": mse_loss,
                "entropy_loss": entropy_loss,
                "code_usage": code_usage,
            }


            if recon_ind < 2:
                loss = mse_loss + entropy_weight * entropy_loss 
            elif recon_ind == 2:
                loss_logs["entropy_loss"] = torch.tensor(0.0, device=mse_loss.device)
                loss_logs["code_usage"] = torch.tensor(0.0, device=mse_loss.device)
                loss = mse_loss


            return loss, loss_logs

        loss1, loss_logs1 = step(recon_ind=0, target_id=0)
        loss2, loss_logs2 = step(recon_ind=1, target_id=1)
        loss3, loss_logs3 = step(recon_ind=2, target_id=1)


        loss_logs = {
            "mse_loss_3": loss_logs3["mse_loss"],
            "mse_loss_2": loss_logs2["mse_loss"],
            "entropy_loss_2": loss_logs2["entropy_loss"],
            "code_usage_2": loss_logs2["code_usage"],
            "mse_loss": loss_logs1["mse_loss"],
            "entropy_loss": loss_logs1["entropy_loss"],
            "code_usage": loss_logs1["code_usage"],
            "entropy_weight": torch.tensor(entropy_weight),
        }


        if batch_idx % self.log_interval == 0:
            if self._state.process_index == 0:
                image = torch.cat([outputs["first"][:4, 0], outputs["target"][:4, 0], outputs["recon"][0][:4], outputs["target"][:4, 1], outputs["recon"][1][:4], outputs["target"][:4, 1], outputs["recon"][1][:4]], dim=0)
                image = torch.clamp(image, min=0, max=1)
                image = make_grid(image, nrow=4, padding=0, pad_value=1.0)
                image = image.permute(1, 2, 0).mul_(255).cpu().numpy()
                image = Image.fromarray(image.astype(np.uint8))
                wandb.log({"recon_images":  wandb.Image(image)}, step=batch_idx)

        if batch_idx % 2000 == 0:
            self.lam.vq_action.reset_usage()
        return outputs, loss1 + loss2 + loss3, loss_logs


    def training_step(self, batch: Dict, batch_idx: int, start_time=0, epoch_len=0) -> Tensor:
        batch["videos"] = batch["videos"].cuda().half()

        #is_main = self._state.is_main_process
        if self.use_identity > 0 and batch_idx > 20000 and batch_idx % self.use_identity == 0:
            with torch.no_grad():
                # Every rank runs the reordered forward so the identity estimate uses
                # the full global batch, not just rank 0's shard.
                outputs, loss, aux_losses = self.shared_step(batch, batch_idx, '000')
                with torch.cuda.amp.autocast(enabled=False):
                    local_mean = outputs["z_q"][0].detach().mean(dim=0, keepdim=True).float()

                # Mean over all ranks. dist.all_reduce sums, so divide by world size.
                global_mean = local_mean.contiguous()
                dist.all_reduce(global_mean, op=dist.ReduceOp.SUM)
                global_mean /= self._state.num_processes

                # SVD only on rank 0 + broadcast: running it per-rank can give
                # sign/ordering differences that would desync the identity buffer.
                with torch.cuda.amp.autocast(enabled=False):
                    if self._state.process_index == 0:
                        U, V, Vh = torch.linalg.svd(global_mean[0, 0])
                        ortho = (U @ Vh).unsqueeze(0).unsqueeze(0)
                    else:
                        ortho = torch.zeros_like(self.lam.identity).float()

                ortho = ortho.contiguous()
                dist.broadcast(ortho, src=0)
                self._state.wait_for_everyone()

                self.lam.identity.data.copy_(ortho.to(self.lam.identity.dtype))
                


        outputs, loss, aux_losses = self.shared_step(batch, batch_idx)

        if batch_idx % self.log_wandb == 0:
            if self._state.process_index == 0 :
                usage = self.lam.vq_action.usage
                wandb.log({**{"train/train_loss": loss.item()}, **{f"train/{k}": aux_losses[k].item() for k in aux_losses.keys()}}, step=batch_idx)
                si = usage.shape[1]
                wandb.log({f"codebook_usage/usage_{k}": sum(usage[k] > 0).item() / si for k in range(self.lam_num_codes)}, step=batch_idx)
                wandb.log({"train/cos_1":outputs["zq_z_cos"][0].item(),"train/cos_2":outputs["zq_z_cos"][1].item()},step=batch_idx)

        if self._state.process_index == 0:
            if batch_idx % 1 == 0 :
                end_time = time.time()
                elapsed = end_time - start_time
                steps_per_sec = 1 / max(elapsed, 1e-6)
                print(
                    f"[Train] step={batch_idx}/{epoch_len}, steps/s={steps_per_sec:.2f}",
                    {
                        **{"train_loss": (loss.item() if hasattr(loss, "item") else loss)},
                        **{f"train/{k}": (aux_losses[k].item() if hasattr(aux_losses[k], "item") else aux_losses[k]) for k in aux_losses.keys()},
                    },
                    flush=True,
                )
        return loss

    @torch.no_grad()
    def test_step(self, batch: Dict, batch_idx: int) -> Tensor:
        outputs, loss, aux_losses = self.shared_step(batch, batch_idx)
        return loss





