
from typing import List, Optional, Tuple, Union
from PIL import Image
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from model.internvl3.internvl3_embedder import InternVL3Embedder
from model.action_head.transformer import SoftPromptedTransformer

from dataset.action_hub import build_action_space

import copy
from dataset.lerobot_dataset_pretrain_mp  import multi_image_get_item,make_conversation,build_transform
from model.lam.modules.lam import SoftDINOLatentActionModel


class RotVLA(nn.Module):
    def __init__(self, config: dict,train=True):
        super().__init__() 
        self.config = config
        self._device = config.get("device", "cuda")
        vlm_name = config.get("vlm_name", "OpenGVLab/InternVL3-1B")
        self.embedder = InternVL3Embedder(model_name=vlm_name, device=self._device,image_size=224)
        self.stage = config.get("stage", "stage3-vla-finetune")
        self.latent_action_model = SoftDINOLatentActionModel(
            in_dim=config["image_channels"],
            model_dim=config["lam_model_dim"],
            latent_dim=config["lam_latent_dim"],
            num_latents=config["lam_num_latents"],
            patch_size=config["lam_patch_size"],
            enc_blocks=config["lam_enc_blocks"],
            dec_blocks=config["lam_dec_blocks"],
            num_codes=config["lam_num_codes"],
            num_heads=config["lam_num_heads"],
            dropout=0.,
            #to_pixel_bool=config["to_pixel"],
            use_identity=config["use_identity"],
        ).to(self._device)
        

        if train : 
                lam_ckpt = torch.load(config["lam_path"],weights_only=False,map_location=self._device)
                if "model" in lam_ckpt.keys():
                    lam_ckpt = lam_ckpt["model"]
                else:
                    lam_ckpt = lam_ckpt["module"]
                new_ckpt = {}
                


                for key in lam_ckpt.keys():

                    #print(key)
                    if 'lam' in key:
                        key2 =copy.deepcopy(key)
                        if ('action_up.' in key or 'to_codebook.' in key):
                            new_ckpt[key.replace("lam.", "")] = lam_ckpt[key]
                            continue
                        
                        elif 'action_latent' in key  and (key2.replace("action_latent", "action_latent_controllable") in lam_ckpt.keys() or 'action_latent_controllable' in key): #对于已有的ckpt
                            if 'action_latent_controllable' in key:
                                #print(key.replace("lam.", "").replace("action_latent_controllable", "action_latent"))
                                new_ckpt[key.replace("lam.", "").replace("action_latent_controllable", "action_latent")] = lam_ckpt[key]
                            else:
                                continue
                        else:
                            new_ckpt[key.replace("lam.", "")] = lam_ckpt[key]

                # for key in lam_ckpt.keys():
                #     new_ckpt[key.replace("lam.", "")] = lam_ckpt[key]

                try:
                    self.latent_action_model.load_state_dict(new_ckpt, strict=True)
                    print('strict load success')
                except Exception as e:
                    print(e)
                    print('strict load failed, try non-strict load')
                    self.latent_action_model.load_state_dict(new_ckpt, strict=False)


        
        
        # action_head_type already set above for LAM loading condition
        self.action_space = build_action_space(config.get("action_space", "ee6d"),norm_path=config.get("norm_path", None), delta_action = config.get("delta_action", False))

            
        horizon = config.get("horizon", 50)
        per_action_dim = config.get("per_action_dim", 24)
        self.state_dim = config.get("state_dim", 24)

        self.per_action_dim = per_action_dim
        self.num_actions = horizon
        self.num_latent_actions = config.get("lam_num_codes", 16)
        # Get VLM output dimension

        embed_dim = config.get("embed_dim", 1024)

        # Build SoftPromptedTransformer
        self.action_head = SoftPromptedTransformer(
            hidden_size=config["embed_dim"],
            multi_modal_input_size=embed_dim,
            num_heads=config["num_heads"],
            mlp_ratio=config.get("mlp_ratio", 4.0),
            num_domains=config.get("num_domains", 1),
            dim_action=per_action_dim,
            dim_propio = self.state_dim,
            dim_latent_action=self.num_latent_actions,
            dim_time=config.get("dim_time", 32),
            max_len_seq=512,
            stage=self.stage,
            pos_emb_scheme=config.get("pos_emb_scheme", "shared"),
        ).to(self._device)


        self.transform = build_transform(self.embedder.image_size)


    def prepare_state(self, state_input: Union[list, torch.Tensor]) -> torch.Tensor:

        if isinstance(state_input, list):
            state_tensor = torch.tensor(state_input)
        elif isinstance(state_input, torch.Tensor):
            state_tensor = state_input
        else:
            raise TypeError("Unsupported state input type")

        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        return state_tensor.to(self._device)

    
    def predict_action(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor = None,
        actions_gt: torch.Tensor = None,
        action_gt_latent = None,
        action_mask: torch.Tensor = None,

    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:


        if actions_gt is None and action_gt_latent is None:
            # Inference mode (diffusion denoising)
            return self._get_action(
                fused_tokens, state=state, action_mask=action_mask,
            )
        else:
            # Training mode
            return self._forward_training(fused_tokens, state=state, actions_gt=actions_gt, action_gt_latent=action_gt_latent, action_mask=action_mask)
    
    def _forward_training(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor = None,
        actions_gt: torch.Tensor = None,
        action_gt_latent: torch.Tensor = None,
        action_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training forward pass. For linear probing, directly predict from VLM features."""
        B = fused_tokens.shape[0]
        device = fused_tokens.device


        t = (torch.rand(1, device=device) + torch.arange(B, device=device) / B) % (1 - 1e-5)


        assert action_gt_latent.ndim == 3, f"action_gt_latent.ndim is {action_gt_latent.ndim} for {B}"

        latent_noise = torch.randn_like(action_gt_latent) #* 2 - 1
        latent_action_noisy = latent_noise * t.view(-1, 1, 1) + action_gt_latent * (1 - t).view(-1, 1, 1)
        latent_action_noisy = latent_action_noisy.to(dtype=fused_tokens.dtype)

        # Default domain_id if not provided
        domain_id = torch.zeros(B, dtype=torch.long, device=device)

        if self.stage == "stage2-vla-pretrain":
            

            _,pred_latent_action = self.action_head(
                domain_id=domain_id,
                vlm_features=fused_tokens,
                latent_action_with_noise=latent_action_noisy,
                t=t,
                num_actions=self.num_actions,
            )

            # Reshape to [B, horizon * per_action_dim] for compatibility with training code
            #pred_velocity = pred_action.view(B, -1)
            pred_latent_action = pred_latent_action.view(B, -1)

            return None, pred_latent_action

        elif self.stage == "stage3-vla-finetune":
            assert actions_gt.ndim == 3

            noise = torch.randn_like(actions_gt) #* 2 - 1
            action_noisy = noise * t.view(-1, 1, 1) + actions_gt * (1 - t).view(-1, 1, 1)
            action_noisy = action_noisy.to(dtype=fused_tokens.dtype)
            
            action_mask = action_mask.to(dtype=action_noisy.dtype, device=device)
            action_noisy = action_noisy * action_mask

            state = state.unsqueeze(1).expand(B, self.num_actions, state.shape[-1]) * action_mask

            state, action_noisy = self.action_space.preprocess(state, action_noisy)

            pred_action,pred_latent_action = self.action_head(
                domain_id=domain_id,
                vlm_features=fused_tokens,
                action_with_noise=action_noisy,
                latent_action_with_noise=latent_action_noisy,
                proprio=state,
                t=t,
                num_actions=self.num_actions,
            )

            # Reshape to [B, horizon * per_action_dim] for compatibility with training code
            pred_velocity = pred_action.view(B, -1)
            pred_latent_action = pred_latent_action.view(B, -1)

            return pred_velocity, pred_latent_action
        
    

    @torch.no_grad()
    def _get_action(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor = None,
        action_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Inference: for linear probing, direct prediction; for diffusion, iterative denoising."""
        self.eval()
        B = fused_tokens.shape[0]
        device = fused_tokens.device

        domain_id = torch.zeros(B, dtype=torch.long, device=device)


        latent_action_noise = torch.randn(B, self.num_latent_actions, self.num_latent_actions, device=device, dtype=fused_tokens.dtype) #* 2 - 1

        num_inference_timesteps = self.config.get("num_inference_timesteps", 32)
        steps = max(1, int(num_inference_timesteps))

        if self.stage == "stage2-vla-pretrain":


            pred_latent_action = torch.zeros_like(latent_action_noise)
            for i in range(steps, 0, -1):
                t = torch.full((B,), i / steps, device=device, dtype=fused_tokens.dtype)


                latent_x_t = latent_action_noise * t.view(-1, 1, 1) + pred_latent_action * (1 - t).view(-1, 1, 1)
                _, pred_latent_action = self.action_head(
                    domain_id=domain_id,
                    vlm_features=fused_tokens,
                    latent_action_with_noise=latent_x_t,
                    t=t,
                    num_actions=self.num_actions,

                )


            pred_latent_action = pred_latent_action.view(B, -1)
            return None, pred_latent_action
        elif self.stage == "stage3-vla-finetune":

            action_noise = torch.randn(B, self.num_actions, self.per_action_dim, device=device, dtype=fused_tokens.dtype)
            if action_mask is  None:
                action_mask = torch.ones_like(action_noise)
                action_mask[:, :,self.action_space.max_action_dim:] = 0
            action_mask = action_mask.to(dtype=action_noise.dtype, device=device)
            action_noise = action_noise * action_mask
            state = state.unsqueeze(1).expand(B, self.num_actions, action_mask.shape[-1]) * action_mask

            pred_action = torch.zeros_like(action_noise)
            pred_latent_action = torch.zeros_like(latent_action_noise)
            for i in range(steps, 0, -1):
                t = torch.full((B,), i / steps, device=device, dtype=fused_tokens.dtype)


                x_t = action_noise * t.view(-1, 1, 1) + pred_action * (1 - t).view(-1, 1, 1)
                latent_x_t = latent_action_noise * t.view(-1, 1, 1) + pred_latent_action * (1 - t).view(-1, 1, 1)
                state, x_t = self.action_space.preprocess(state, x_t)
                pred_action, pred_latent_action = self.action_head(
                    domain_id=domain_id,
                    vlm_features=fused_tokens,
                    action_with_noise=x_t,
                    latent_action_with_noise=latent_x_t,
                    proprio=state,
                    t=t,
                    num_actions=self.num_actions,

                )

                pred_action = pred_action * action_mask


            pred_action = pred_action.view(B, -1)
            pred_latent_action = pred_latent_action.view(B, -1)
            return pred_action, pred_latent_action
        else:
            raise ValueError(f"Invalid stage: {self.stage}")


        # action_noise = torch.randn(B, self.num_actions, self.per_action_dim, device=device, dtype=fused_tokens.dtype)
        
        # if action_mask is  None:
        #     action_mask = torch.ones_like(action_noise)
        # if state is None:
        #     state = torch.zeros(B, self.per_action_dim, device=device, dtype=fused_tokens.dtype)
        # if action_mask is not None:
            
        #     action_mask = action_mask.to(dtype=action_noise.dtype, device=device)
        #     action_noise = action_noise * action_mask
        #     state = state.unsqueeze(1).expand(B, self.num_actions, action_mask.shape[-1]) * action_mask

        # latent_action_noise = torch.randn(B, self.num_latent_actions, self.num_latent_actions, device=device, dtype=fused_tokens.dtype) #* 2 - 1

        

        # pred_action = torch.zeros_like(action_noise)
        # pred_latent_action = torch.zeros_like(latent_action_noise)
        # for i in range(steps, 0, -1):
        #     t = torch.full((B,), i / steps, device=device, dtype=fused_tokens.dtype)
            

        #     x_t = action_noise * t.view(-1, 1, 1) + pred_action * (1 - t).view(-1, 1, 1)
        #     latent_x_t = latent_action_noise * t.view(-1, 1, 1) + pred_latent_action * (1 - t).view(-1, 1, 1)
        #     state, x_t = self.action_space.preprocess(state, x_t,max_state_dim=self.max_state_dim)
        #     pred_action, pred_latent_action = self.action_head(
        #         domain_id=domain_id,
        #         vlm_features=fused_tokens,
        #         action_with_noise=x_t,
        #         latent_action_with_noise=latent_x_t,
        #         proprio=state,
        #         t=t,
        #         num_actions=self.num_actions,

        #     )
            
        #     if action_mask is not None:
        #         pred_action = pred_action * action_mask
            
          
        # pred_action = pred_action.view(B, -1)
        # pred_latent_action = pred_latent_action.view(B, -1)
        # return pred_action, pred_latent_action
    


    @torch.no_grad()
    def run_inference(
        self,
        images: List[Union[Image.Image, torch.Tensor]],
        prompt: str,
        state_input: Union[list, torch.Tensor],
        action_mask: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:


        
        
        prompt = make_conversation(prompt=prompt)
        results = multi_image_get_item(
            image_tensors=images,
            prompt=prompt,
            transform=self.transform,
            image_size=224, 
            text_tokenizer=self.embedder.tokenizer,
            num_image_token=64, # 需要改
        )

        
        vlm_outputs = self.common_process(pixel_values=results["pixel_values"].to(self._device),
            input_ids=results["input_ids"].unsqueeze(0).to(self._device),
            attention_mask=results["attention_mask"].unsqueeze(0).to(self._device),
            position_ids=results["position_ids"].unsqueeze(0).to(self._device),
            image_flags=results["image_flags"],
            labels=results["labels"].unsqueeze(0).to(self._device),
            output_hidden_states=True,)

        fused_tokens = vlm_outputs.hidden_states[-1]
        
        state_tensor = self.prepare_state(state_input)  
        
        action, _ = self.predict_action(
            fused_tokens, state_tensor,
            actions_gt=None, action_gt_latent=None, action_mask=action_mask,
        )
        return action

    def forward(self, fused_tokens, state=None, actions_gt=None,action_gt_latent=None, action_mask=None):
   

        return self.predict_action(fused_tokens, state, actions_gt, action_gt_latent, action_mask)

    def common_process(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = True,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = False,
    ) -> CausalLMOutputWithPast:
        # Align with original InternVL implementation style
      
        #device = self.embedder.device
        image_flags = image_flags.squeeze(-1)
        input_embeds = self.embedder.model.language_model.get_input_embeddings()(input_ids).clone()
        vit_embeds = self.embedder.model.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(-1, C)

        input_ids = input_ids.reshape(-1)

        selected = input_ids == self.embedder.img_context_token_id
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            raise ValueError(
                f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, vit_embeds.shape={vit_embeds.shape}"
            )

        input_embeds = input_embeds.reshape(B, N, C)

        vlm_outputs = self.embedder.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=return_dict,
            labels=labels,
            output_hidden_states=output_hidden_states,
        )
        return vlm_outputs


    def _freeze_module(self, module: nn.Module, name: str):
        for p in module.parameters():
            p.requires_grad = False

    def set_finetune_flags(self):
        config = self.config  
        if not config.get("finetune_vlm", False):
            self._freeze_module(self.embedder, "VLM (InternVL3)")
        else:
            print("Finetuning VLM (InternVL3)...")

        if not config.get("finetune_action_head", False):
            self._freeze_module(self.action_head, "Action Head")
        else:
            print("Finetuning Action Head...")
        
        if self.latent_action_model is not None:
            self._freeze_module(self.latent_action_model, "Latent Action Model")
