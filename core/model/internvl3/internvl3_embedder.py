# model/internvl3/internvl3_embedder.py
import torch
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from dataset.utils import IMG_CONTEXT_TOKEN

from torch import nn





class InternVL3Embedder(nn.Module):
    def __init__(self, model_name="OpenGVLab/InternVL3-1B", image_size=448, device="cuda"):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.max_text_length = 1024  # InternVL3 supports up to 1024 tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name,add_eos_token=False, trust_remote_code=True, use_fast=False)

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_flash_attn=True,
            low_cpu_mem_usage=True,
            _fast_init=False,
        ).to(self.device) 
        
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
