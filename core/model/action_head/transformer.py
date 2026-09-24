# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from __future__ import annotations

import math
from functools import partial
from typing import Final, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------- Small utils ----------------------------------

def _to_2tuple(x) -> Tuple:
    """Minimal replacement for timm.layers.to_2tuple."""
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        t = tuple(x)
        return (t[0], t[1]) if len(t) >= 2 else (t[0], t[0])
    return (x, x)


def _has_sdp_attention() -> bool:
    """Check if we can use PyTorch fused scaled_dot_product_attention."""
    return hasattr(F, "scaled_dot_product_attention")


# ---------------------------------- MLP --------------------------------------

class Mlp(nn.Module):
    """
    MLP used in ViT-style blocks.

    Supports Linear or 1x1 Conv 'linear_layer' for token/channel mixing.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        norm_layer: type[nn.Module] | None = None,
        bias: bool | Tuple[bool, bool] = True,
        drop: float | Tuple[float, float] = 0.0,
        use_conv: bool = False,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = _to_2tuple(bias)
        drop_probs = _to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect [B, T, C] for Linear variant; caller is responsible for shapes.
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# -------------------------------- Attention ----------------------------------

class Attention(nn.Module):
    """
    Multi-Head Self-Attention with optional fused SDPA fallback.

    If PyTorch provides `scaled_dot_product_attention`, it will be used
    (usually faster and more stable); otherwise we use a manual implementation.
    """

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = _has_sdp_attention()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape [B, T, C]
            Input sequence.
        attn_mask : Tensor, shape [B, T, T] or [T, T], optional
            Attention mask, following `torch.nn.functional.scaled_dot_product_attention`
            semantics. If boolean, True means the position CAN be attended to and False
            means it is masked out. If floating point, it is added to the attention
            scores (use -inf to mask out).

        Returns
        -------
        Tensor, shape [B, T, C]
            Output sequence after MHSA + projection.
        """
        B, T, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, T, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)  # 3 x [B, H, T, Dh]
        )
        q, k, v = qkv.unbind(0)  # each: [B, H, T, Dh]
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0).expand(B, -1, -1)
                attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )  # [B, H, T, Dh]
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)        # [B, H, T, T]
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)
                elif attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
                # Match SDPA semantics: boolean True = attend, False = mask out;
                # floating point mask is added to the scores.
                if attn_mask.dtype == torch.bool:
                    attn = attn.masked_fill(~attn_mask, float('-inf'))
                else:
                    attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v                           # [B, H, T, Dh]

        x = x.transpose(1, 2).reshape(B, T, C)     # [B, T, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ------------------------------- Utilities -----------------------------------

def basic_init(module: nn.Module) -> None:
    """
    Apply a basic initialization scheme to Linear layers.

    - Weight: Xavier uniform initialization.
    - Bias: Set to zero.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 100) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Parameters
    ----------
    t : torch.Tensor
        Shape [B]. Each element is a timestep index, may be fractional.
    dim : int
        Dimensionality of the output embedding.
    max_period : int, default=100
        Controls the minimum frequency of the sinusoids.

    Returns
    -------
    torch.Tensor
        Shape [B, dim]. Sinusoidal embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=t.dtype, device=t.device)
        / half
    )
    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ------------------------------- Core Layers ----------------------------------

class DomainAwareLinear(nn.Module):
    """
    Linear layer with domain-conditioned parameters (per-sample).

    Each domain has its own weight and bias vectors, stored in embeddings.
    """

    def __init__(self, input_size: int, output_size: int, num_domains: int = 20) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Embedding(num_domains, output_size * input_size)
        self.bias = nn.Embedding(num_domains, output_size)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x: torch.Tensor, domain_id: torch.LongTensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            [B, I] or [B, T, I]
        domain_id : LongTensor
            [B], domain indices.

        Returns
        -------
        Tensor
            [B, O] or [B, T, O]
        """
        B = domain_id.shape[0]
        squeeze_T = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_T = True
        W = self.fc(domain_id).view(B, self.input_size, self.output_size)
        b = self.bias(domain_id).view(B, self.output_size)
        y = torch.matmul(x, W) + b.view(B, 1, self.output_size)
        if squeeze_T:
            y = y.squeeze(1)
        return y


class TransformerBlock(nn.Module):
    """
    Standard Transformer block (pre-LN): LN → MHSA → residual, LN → MLP → residual.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            drop=0.1,
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, [B, T, H]
        attn_mask : Tensor, optional
            Attention mask to pass to attention layer.

        Returns
        -------
        Tensor, [B, T, H]
        """
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------- Main Model ---------------------------------------

class SoftPromptedTransformer(nn.Module):
    """
    Multi-modal, domain-aware Transformer with optional soft prompts.

    See parameter and forward I/O descriptions inside the docstrings.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        multi_modal_input_size: int = 768,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_domains: int = 20,
        dim_action: int = 20,
        dim_propio: int = 20,
        dim_time: int = 32,
        max_len_seq: int = 512,
        dim_latent_action: int = 16,
        stage: str = "stage3-vla-finetune",
        pos_emb_scheme: str = "shared",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dim_action = dim_action
        self.dim_latent_action = dim_latent_action
        self.dim_time = dim_time
        self.stage = stage
        # Positional-embedding scheme:
        #   - "split"  : two independent pools (`pos_emb` for the [latent, vlm]
        #                context from position 0, `action_pos_emb` for the action
        #                segment). Lets the action horizon change between stage2
        #                and stage3 without shifting the context positions.
        #   - "shared" : single `pos_emb` pool with the legacy num_actions-offset
        #                indexing (matches Evo-1 and the old RotVLA/VLA-Pretrain
        #                checkpoints, which only contain `action_head.pos_emb`).
        assert pos_emb_scheme in ("split", "shared"), \
            f"pos_emb_scheme must be 'split' or 'shared', got {pos_emb_scheme!r}"
        self.pos_emb_scheme = pos_emb_scheme
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )

        self.vlm_proj = nn.Linear(multi_modal_input_size, hidden_size)
        #self.aux_visual_proj = nn.Linear(multi_modal_input_size, hidden_size)

        self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size), requires_grad=True)
        nn.init.normal_(self.pos_emb, std=0.02)
        # The second pool only exists in the "split" scheme. Skipping it in
        # "shared" mode keeps the state_dict identical to the legacy checkpoints
        # so they load with strict=True (no spurious `action_pos_emb` key).
        if self.pos_emb_scheme == "split":
            self.action_pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size), requires_grad=True)
            nn.init.normal_(self.action_pos_emb, std=0.02)

        self.norm = nn.LayerNorm(hidden_size)
        self.action_encoder = DomainAwareLinear(
            dim_action + dim_time + dim_propio, hidden_size, num_domains=num_domains
        ) # 48 64 
        self.action_decoder = DomainAwareLinear(hidden_size, dim_action, num_domains=num_domains)

        self.latent_action_encoder = DomainAwareLinear(
            dim_latent_action + dim_time , hidden_size, num_domains=num_domains
        )
        self.latent_action_decoder = DomainAwareLinear(
            hidden_size , dim_latent_action, num_domains=num_domains
        )
        
        self.apply(basic_init)

    def forward(
        self,
        domain_id: torch.LongTensor,
        vlm_features: torch.Tensor,
        latent_action_with_noise: torch.Tensor,
        t: torch.Tensor,
        num_actions: int,
        action_with_noise: torch.Tensor = None,
        proprio: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Inputs
        ------
        domain_id : [B]
        vlm_features : [B, T_vlm, D]
        action_with_noise : [B, T_action, dim_action]
        proprio : [B, dim_propio]
        t : [B]

        Returns
        -------
        Tensor
            Predicted actions, [B, T_action, dim_action]
        """
        

        B, num_latent_actions = latent_action_with_noise.shape[:2]
        # Encode (action + proprio + time) → tokens
        time_emb = timestep_embedding(t, self.dim_time).to(vlm_features.dtype)                    # [B, dim_time]
        latent_time_tokens = time_emb.unsqueeze(1).expand(B, num_latent_actions, self.dim_time)
        latent_action_tokens = torch.cat([latent_action_with_noise, latent_time_tokens], dim=-1) # [B, T_action, 80] # 24 + 24 + 32
        latent_x = self.latent_action_encoder(latent_action_tokens, domain_id)
        
        if self.stage == "stage2-vla-pretrain":


            x = torch.cat([latent_x, self.vlm_proj(vlm_features)], dim=1)


            seq_len = x.shape[1]
            if self.pos_emb_scheme == "shared":
                # Legacy scheme: single pool, latent/vlm shifted by num_actions so
                # they occupy the same absolute positions as in stage3
                # ([action, latent, vlm] indexed from 0). num_actions is the
                # action horizon reserved for stage3 even though there are no
                # action tokens here.
                if seq_len + num_actions > self.pos_emb.shape[1]:
                    raise ValueError(
                        f"Sequence length {seq_len}+num_actions {num_actions} exceeds "
                        f"max_len_seq={self.pos_emb.shape[1]}."
                    )
                x = x + self.pos_emb[:, num_actions:seq_len + num_actions, :]
            else:
                if seq_len > self.pos_emb.shape[1]:
                    raise ValueError(
                        f"Sequence length {seq_len} exceeds max_len_seq={self.pos_emb.shape[1]}."
                    )
                # [latent, vlm] context is indexed from position 0 — identical to the
                # context segment in stage3, so the learned representation transfers
                # regardless of the (absent here) action horizon.
                x = x + self.pos_emb[:, :seq_len, :]

            # Transformer backbone
            for block in self.blocks:
                x = block(x)

            return None, self.latent_action_decoder(self.norm(x[:, :num_latent_actions]), domain_id)

        elif self.stage == "stage3-vla-finetune":

            
            time_tokens = time_emb.unsqueeze(1).expand(B, num_actions, self.dim_time)
            action_tokens = torch.cat([action_with_noise, proprio, time_tokens], dim=-1) # [B, T_action, 80] # 24 + 24 + 32

            action_x = self.action_encoder(action_tokens, domain_id)            # [B, num_actions, H]

            if self.pos_emb_scheme == "shared":
                # Legacy scheme: single pool over the whole [action, latent, vlm]
                # sequence from position 0 (matches Evo-1 two_action + the legacy
                # RotVLA/VLA-Pretrain checkpoints). The latent/vlm block therefore
                # sits at positions [num_actions:], exactly where the num_actions
                # offset in the "shared" stage2 branch placed it during pretraining.
                x = torch.cat([action_x, latent_x, self.vlm_proj(vlm_features)], dim=1)
                seq_len = x.shape[1]
                if seq_len > self.pos_emb.shape[1]:
                    raise ValueError(
                        f"Sequence length {seq_len} exceeds max_len_seq={self.pos_emb.shape[1]}."
                    )
                x = x + self.pos_emb[:, :seq_len, :]
            else:
                # [latent, vlm] context segment — indexed from position 0 with the
                # shared `pos_emb`, matching stage2 exactly.
                context_x = torch.cat([latent_x, self.vlm_proj(vlm_features)], dim=1)
                context_len = context_x.shape[1]
                if context_len > self.pos_emb.shape[1]:
                    raise ValueError(
                        f"Context length {context_len} exceeds max_len_seq={self.pos_emb.shape[1]}."
                    )
                if num_actions > self.action_pos_emb.shape[1]:
                    raise ValueError(
                        f"num_actions {num_actions} exceeds max_len_seq={self.action_pos_emb.shape[1]}."
                    )
                # Action segment gets its own positional pool, so the horizon can
                # differ from stage2 without shifting the context positions.
                action_x = action_x + self.action_pos_emb[:, :num_actions, :]
                context_x = context_x + self.pos_emb[:, :context_len, :]

                x = torch.cat([action_x, context_x], dim=1)
                seq_len = x.shape[1]

            # Transformer backbone
            attn_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
            attn_mask[num_actions:num_actions+num_latent_actions, :num_actions] = False

            for block in self.blocks:
                x = block(x, attn_mask=attn_mask)

            # Decode only the action segment
            return self.action_decoder(self.norm(x[:, :num_actions]), domain_id),self.latent_action_decoder(self.norm(x[:, num_actions:num_actions+num_latent_actions]), domain_id)