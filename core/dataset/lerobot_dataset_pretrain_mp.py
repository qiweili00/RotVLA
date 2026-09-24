#use lerobot_dataset_pretrain_mp.py for multithreading load dataset
import os

import random
import numpy as np
from typing import List, Union, Dict, Any, Sequence
import transformers

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.functional import to_pil_image

import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import Sampler

import logging
import robosuite.utils.transform_utils as Trans


from dataset.utils import IMAGENET_MEAN, IMAGENET_STD
from dataset.utils import get_conv_template
from dataset.utils import IMG_CONTEXT_TOKEN,IMG_START_TOKEN,IMG_END_TOKEN,IGNORE_TOKEN_ID
from dataset.action_hub import build_action_space


from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)

from dataset import OXE_NAMED_MIXTURES

EPS = 1e-6

def Mat_to_Rotate6D( R: np.ndarray) -> np.ndarray:
    if R.ndim == 2:
        return np.concatenate([R[:3, 0], R[:3, 1]], axis=-1)
    elif R.ndim == 3:
        return np.concatenate([R[:, :3, 0], R[:, :3, 1]], axis=-1)
    else:
        raise ValueError("Rotation matrix must be (...,3,3)")

def AxisAngle_to_Rotate6D( aa: np.ndarray) -> np.ndarray:
    if aa.ndim == 1:
        return Mat_to_Rotate6D(Trans.quat2mat(Trans.axisangle2quat(aa)))
    else:
        raise ValueError("Only 1D axis-angle supported here")


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def make_conversation(prompt: str, conversation_type: int = 0) -> str:
    if conversation_type == 0:
        return f"What action should the robot take to {prompt}?"
    elif conversation_type == 1:
        return f"What action should the robot take to {prompt}?"
    elif conversation_type == 2:
        return f"{prompt}"
    else:
        print(f"Conversation Type {conversation_type} is not implemented.")
        raise NotImplementedError()



def preprocess_internvl(
    template_name,
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    num_image_token_list: list,
    text_only: bool = False,
    group_by_length: bool = False,
    use_packed_ds: bool = False,
    ds_name: str = None,
    num_image: int = 1,
) -> Dict:
    assert len(sources) == 1, "process only the first conversations"
    conversations = sources[0]

    if conversations[0]["from"] == "system":
        system_prompt = conversations[0]["value"]
        conversations = conversations[1:]  # remove system prompt
    else:
        conv = get_conv_template(template_name)
        system_prompt = conv.system_message
        # system_prompt = None

    if not text_only:
        new_conversations = []
        current_image_idx = 0
        for conversation in conversations:
            if conversation["from"] == "human":
                image_cnt = conversation["value"].count("<image>")
                for i in range(image_cnt):
                    if current_image_idx == num_image:
                        break
                    image_tokens = (
                        f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[current_image_idx]}{IMG_END_TOKEN}"
                    )
                    conversation["value"] = conversation["value"].replace("<image>", image_tokens, 1)
                    current_image_idx += 1
            new_conversations.append(conversation)
        conversations = new_conversations
        assert current_image_idx == num_image, f"{current_image_idx} != {num_image}"

    batches, roles = [], []
    if system_prompt is not None:
        batches.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")
        roles.append("system")
    for conversation in conversations:
        if conversation["from"] == "human":
            batches.append(f'<|im_start|>user\n{conversation["value"]}<|im_end|>\n')
            roles.append("human")
        elif conversation["from"] == "gpt":
            batches.append(f'<|im_start|>assistant\n{conversation["value"]}<|im_end|>\n')
            roles.append("gpt")
        else:
            raise NotImplementedError
    add_bos_token = getattr(tokenizer, 'add_bos_token', False)
    if add_bos_token:  # for InternLM series
        batches[0] = tokenizer.bos_token + batches[0]

    # Tokenize conversations
    input_ids = tokenizer(
        batches,
        return_tensors="np",
        padding=False,
        max_length=tokenizer.model_max_length,
        truncation=False,
    ).input_ids

    if add_bos_token:  # for InternLM series
        input_ids = [item[1:] for item in input_ids]

    final_input_ids, final_targets = [], []
    ignore_ids = tokenizer("<|im_start|>assistant\n", return_tensors="np").input_ids[0]
    ignore_len = ignore_ids.shape[0] - 1 if add_bos_token else ignore_ids.shape[0]

    for role, input_id in zip(roles, input_ids):
        if role == "system" or role == "human":
            final_targets.append(np.full(input_id.shape, IGNORE_TOKEN_ID))  # ignore
        elif role == "gpt":
            target = input_id.copy()
            target[:ignore_len] = IGNORE_TOKEN_ID  # ignore loss for `<|im_start|>assistant\n`
            target[-1:] = IGNORE_TOKEN_ID  # ignore loss for `\n`
            final_targets.append(target)
        else:
            raise NotImplementedError
        final_input_ids.append(input_id)
    input_ids = torch.tensor(np.concatenate(final_input_ids))[: tokenizer.model_max_length]
    targets = torch.tensor(np.concatenate(final_targets))[: tokenizer.model_max_length]

    padding = False if group_by_length or use_packed_ds else True
    if padding:
        current_length = input_ids.size(0)
        padding_length = tokenizer.model_max_length - current_length
        input_ids = F.pad(input_ids, (0, padding_length), value=tokenizer.pad_token_id)
        targets = F.pad(targets, (0, padding_length), value=IGNORE_TOKEN_ID)

    input_ids = input_ids.unsqueeze(0)
    targets = targets.unsqueeze(0)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


# === Aspect Ratio Handling ===
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff and area > 0.5 * image_size**2 * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=1, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images



def multi_image_get_item(
    image_tensors,
    prompt,
    transform,
    image_size,
    text_tokenizer,
    num_image_token,
):
    images, num_tiles = [], []
    num_image = 0
    pixel_values_list = []
    for i, image in enumerate(image_tensors):
        if isinstance(image, torch.Tensor):
            image = to_pil_image(image)
        tiles = dynamic_preprocess(image, image_size=image_size)
        tile_tensors = torch.stack([transform(t) for t in tiles])  # (T_i, 3, 448, 448)
        pixel_values_list.append(tile_tensors)
        num_image += 1

    pixel_values = torch.cat(pixel_values_list, dim=0).to(dtype=torch.bfloat16)
    num_patches = pixel_values.size(0)
    num_tiles = [pv.shape[0] for pv in pixel_values_list]


    # Preprocess the conversations and generate the return dictionary
    num_image_tokens = [num_image_token * num_tile for num_tile in num_tiles]
    conversation = [
        {"from": "human", "value": f"{'<image>'*num_image}{prompt}"},
        {"from": "gpt", "value": ""},
    ]
    ret = preprocess_internvl(
        "internvl2_5",
        [conversation],
        text_tokenizer,
        num_image_tokens,
        num_image=num_image,
        group_by_length=True,
    )

    # Calculate position_ids for packed dataset
    position_ids = ret["attention_mask"].long().cumsum(-1) - 1
    position_ids.masked_fill_(ret["attention_mask"] == 0, 1)
    image_end_token_id = text_tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
    assert (ret["input_ids"][0] == image_end_token_id).sum() == num_image, "image tokens are truncated"

    # Create the final return dictionary
    final_ret = dict(
        input_ids=ret["input_ids"][0],
        labels=ret["labels"][0],
        attention_mask=ret["attention_mask"][0],
        position_ids=position_ids[0],
        pixel_values=pixel_values,
        image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
    )
    return final_ret



class MixtureDataset(Dataset):
    def __init__(
        self, 
        datasets: Sequence[Dataset], 
        datasets_name: Sequence[str], 
        datasets_meta: Sequence[dict], 
        datasets_weights: dict = None,
        datasets_weights_raw: dict = None,
        num_eval_episodes: int = None,
        stage: str = "stage2-vla-pretrain",
        shuffle: bool = True,
    ):
        self.datasets = datasets
        self.datasets_name = datasets_name
        self.meta = datasets_meta
        self.num_episodes = [info["num_episodes"] for info in datasets_meta]
        self.arm_keys = [info["robot_type"] for info in datasets_meta]
        self.num_frames = [info["num_frames"] for info in datasets_meta]
        self.stage = stage
        
        assert stage in [
            "stage1-latent-pretrain",
            "stage2-vla-pretrain",
            "stage3-vla-finetune",
        ]
        self.datasets_weights_raw = datasets_weights_raw
        self._compute_len()
        self._get_weights(datasets_weights)

        assert len(self.flat_sample_map) == len(self.sample_weights)

        if shuffle:
            # shuffle the flat_sample_map and sample_weights in the same order
            indices = np.random.permutation(len(self.flat_sample_map))
            self.flat_sample_map = [self.flat_sample_map[i] for i in indices]
            self.sample_weights = self.sample_weights[indices]


    def __len__(self):
        return len(self.flat_sample_map)

    def __getitem__(self, index):
        if index < 0 or index >= len(self.flat_sample_map):
            raise IndexError(f"Index {index} is out of bounds for the dataset.")
        dataset_idx, sample_idx = self.flat_sample_map[index]
        while True:
            try :
                sample = self.datasets[dataset_idx][sample_idx]
                break
            except Exception as e:
                print(f"Error getting item {index} from dataset {dataset_idx} name: {self.datasets_name[dataset_idx]}: {e}")
                index = (index + 100) % len(self.flat_sample_map)
                dataset_idx, sample_idx = self.flat_sample_map[index]

        return (dataset_idx, sample)


    def _compute_len(self):
        self.all_sample_indices = []

        for i, (actual_ds, meta) in enumerate(zip(self.datasets, self.meta)):
            # Only stage1 uses limited sampling via num_indices, other stages use all data
            # num_indices = meta.get("num_indices") if self.stage.startswith("stage1") else None
            num_indices = None
            assert isinstance(actual_ds, LeRobotDataset)
            

            _from = actual_ds.meta.episodes["dataset_from_index"]
            _to = actual_ds.meta.episodes["dataset_to_index"]
            indices = self._sample_indices(_from, _to, num_indices,random_pad=True, dataset_name=self.datasets_name[i],weights=self.datasets_weights_raw.get(self.datasets_name[i],1.0))
            self.all_sample_indices.append(indices)

        self.flat_sample_map = self._create_flat_sample_map()

    def _create_flat_sample_map(self):
        flat_map = []
        for dataset_idx, sample_group in enumerate(self.all_sample_indices):
            if isinstance(sample_group, list) and len(sample_group) > 0 and isinstance(sample_group[0], list):
                for sub_group in sample_group:
                    for tensor in sub_group:
                        for i in range(tensor.numel()):
                            flat_map.append((dataset_idx, tensor[i].item()))
            elif isinstance(sample_group, list) and len(sample_group) > 0:
                for tensor in sample_group:
                    for i in range(tensor.numel()):
                        flat_map.append((dataset_idx, tensor[i].item()))
            elif isinstance(sample_group, torch.Tensor):
                for i in range(sample_group.numel()):
                    flat_map.append((dataset_idx, sample_group[i].item()))
        
        return flat_map

    def _sample_indices(self, start, end, num_frames, random_pad=False, dataset_name=None,weights=None):
        all_indices = []
        for _start, _end in zip(start, end):

            if 'robomind' in dataset_name.lower() or 'robocoin' in dataset_name.lower():
                delta = _end - _start
                _start = _start + (delta // 10)
                _end = _end - (delta // 10)
            frame_count = _end - _start

            if self.stage == 'stage3-vla-finetune':
                target_frames = frame_count
            else:
                target_frames = max(1, int((frame_count*weights)))

            
            #target_frames = frame_count
            if frame_count >= target_frames:
                indices = torch.linspace(_start, _end - 1, steps=target_frames).long()
            else:
                if random_pad:
                    pad_size = target_frames - frame_count
                    indices = torch.arange(_start, _end)
                    pad_indices = indices[torch.randint(0, frame_count, (pad_size,))]
                    indices = torch.cat([indices, pad_indices])
                    indices = indices[torch.randperm(target_frames)]
                else:
                    indices = torch.arange(_start, _end)

            all_indices.append(indices)

        return all_indices

    def _get_weights(self, datasets_weights: dict):
        """Assign normalized sampling weights to each sample."""
        self.sample_weights = []
        self.datasets_weight_map = {}
        self.original_weights = {}  # Store original weights before normalization
        self.dataset_lengths = {}  # Store original dataset lengths

        for ds_name, sample_group in zip(self.datasets_name, self.all_sample_indices):
            if isinstance(sample_group, list) and isinstance(sample_group[0], torch.Tensor):
                flattened = sample_group
            elif isinstance(sample_group, list):
                flattened = [x for sublist in sample_group for x in sublist]
            else:
                flattened = [sample_group]

            num_indices = sum(x.numel() for x in flattened)
            weight = 1.0 #datasets_weights.get(ds_name, 1.0)
            
            # Store original weight and dataset length
            self.original_weights[ds_name] = weight
            self.dataset_lengths[ds_name] = num_indices

            if ds_name not in self.datasets_weight_map:
                self.datasets_weight_map[ds_name] = [weight] * num_indices
            else:
                self.datasets_weight_map[ds_name].extend([weight] * num_indices)
            
            self.sample_weights.extend([weight] * num_indices)

        total_weights = sum(self.sample_weights)
        for k, v in self.datasets_weight_map.items():
            self.datasets_weight_map[k] = sum(v) / total_weights

        self.sample_weights = np.array(self.sample_weights, dtype=np.float32)
        self.sample_weights = self.sample_weights / np.sum(self.sample_weights)

    
    def to_log_string(self):
        """String representation without color codes, suitable for logging."""
        lines = [
            "=" * 80,
            "Dataset Mixture Statistics",
            "=" * 80
        ]
        
        if len(self.datasets_weight_map) > 0:
            max_key_len = max(len(k) for k in self.datasets_weight_map.keys()) + 2
            
            # Calculate total original length for length-based normalization
            total_original_len = sum(self.dataset_lengths.values())
            
            # Header
            lines.append(f"{'Dataset':<{max_key_len}} | {'Weight':>10} | {'Weight Norm%':>12} | {'Length Norm%':>12} | {'Sampled Len':>15}")
            lines.append("-" * (max_key_len + 65))
            
            # Dataset details
            for k in self.datasets_weight_map.keys():
                original_weight = self.original_weights.get(k, 0.0)
                weight_normalized_pct = self.datasets_weight_map[k] * 100  # Based on weights
                original_len = self.dataset_lengths.get(k, 0)
                # Length-based normalization (without considering weights)
                length_normalized_pct = (original_len / total_original_len * 100) if total_original_len > 0 else 0.0
                # Count sampled samples for this dataset
                dataset_idx = self.datasets_name.index(k)
                sampled_len = sum(1 for dataset_idx_map, _ in self.flat_sample_map 
                                 if dataset_idx_map == dataset_idx)
                
                lines.append(
                    f"{k:<{max_key_len}} | {original_weight:>10.2f} | "
                    f"{weight_normalized_pct:>11.2f}% | {length_normalized_pct:>11.2f}% | {sampled_len:>15}"
                )
        else:
            max_key_len = 20
        
        lines.append("-" * (max_key_len + 65))
        lines.append(f"{'Total episodes':<{max_key_len}} : {sum(self.num_episodes):>10.0f}")
        lines.append(f"{'Total frames':<{max_key_len}} : {sum(self.num_frames):>10.0f}")
        lines.append(f"{'Total samples':<{max_key_len}} : {len(self.flat_sample_map):>10.0f}")
        lines.append(f"{'Stage':<{max_key_len}} : {self.stage:>10s}")
        lines.append("=" * 80)
        
        return "\n".join(lines)


class LeRobotDatasetAdapter(Dataset):
    """Adapter to convert lerobot dataset format to expected format"""
    def __init__(self, mixture_dataset: 'MixtureDataset', tokenizer, image_transform, image_transform_lam,aug_transform, vla_image_resolution, num_image_token=256,data_info=None,use_augmentation=False,stage='stage2-vla-pretrain',max_views=1, arm2stats_dict=None, state_dim=24, per_action_dim=24, horizon=16, action_space_type='eepose',delta_action=False,norm_path=None):
        self.mixture_dataset = mixture_dataset
        self.dataset_name = self.mixture_dataset.datasets_name
        self.arm_keys = self.mixture_dataset.arm_keys
        self.tokenizer = tokenizer
        self.aug_transform = aug_transform
        self.image_transform = image_transform
        self.image_transform_lam = image_transform_lam
        self.vla_image_resolution = vla_image_resolution
        self.num_image_token = num_image_token
        self.data_info = data_info
        self.use_augmentation = use_augmentation
        self.stage = stage
        self.max_views = max_views
        self.arm2stats_dict = arm2stats_dict
        self.state_dim = state_dim
        self.per_action_dim = per_action_dim
        self.delta_action = delta_action

        self.action_space= build_action_space(action_space_type,norm_path=norm_path,delta_action=delta_action)
        if stage == "stage3-vla-finetune" :

            self.state_key = self.action_space.get_state_key()
            self.action_key = self.action_space.get_action_key()
        assert stage in [
            "stage1-latent-pretrain",
            "stage2-vla-pretrain",
            "stage3-vla-finetune",
        ]
        
        

        
        # Build transform for multi_image_get_item (includes normalization)
        self.transform = build_transform(self.vla_image_resolution)
    
    def _pad_tensor(
        self, 
        source_tensor: torch.Tensor, 
        max_dim: int
    ) -> (torch.Tensor, torch.Tensor):

        source_dim = source_tensor.shape[-1]
        
        if source_tensor.dim() > 1:
            padded_shape = (*source_tensor.shape[:-1], max_dim)
        else:
            padded_shape = (max_dim,)

        padded_tensor = torch.zeros(padded_shape, dtype=source_tensor.dtype, device=source_tensor.device)
        mask = torch.zeros(padded_shape, dtype=torch.bool, device=source_tensor.device)

        data_slice = (..., slice(0, source_dim))
        
        padded_tensor[data_slice] = source_tensor
        mask[data_slice] = True
            
        return padded_tensor, mask

    def __len__(self):
        return len(self.mixture_dataset)
    
    def __getitem__(self, index):
        dataset_idx, sample = self.mixture_dataset[index]
        # Extract data from lerobot sample format
        # Lerobot dataset returns dict with keys like 'observation', 'action', etc.
        img_data_list = []
        for num_image in range(self.max_views):
            
            
            if f"image_{num_image+1}" in self.data_info[self.dataset_name[dataset_idx]]["view_map"].keys():
                observation = self.data_info[self.dataset_name[dataset_idx]]["view_map"][f"image_{num_image+1}"]
                img_data_list.append(sample[observation])

        pil_images = []

        for img_tensor in img_data_list:
            if len(img_tensor.shape) == 4:
                if isinstance(img_tensor[0], torch.Tensor):
                    pil_img = to_pil_image(img_tensor[0])
                    pil_images.append(pil_img)
                else:
                    pil_images.append(img_tensor[0])         
            else:
                if isinstance(img_tensor, torch.Tensor):
                    pil_img = to_pil_image(img_tensor)
                    pil_images.append(pil_img)
                else:
                    pil_images.append(img_tensor)

        if self.stage == 'stage2-vla-pretrain':

            if self.use_augmentation:
           
                img_data = [
                    self.aug_transform(pil_image) if random.random() < 0.5 else self.image_transform(pil_image)
                    for pil_image in pil_images 
                ]
            else:
            
                img_data = [self.image_transform(pil_image) for pil_image in pil_images]

            while len(img_data) < self.max_views:
           
                dummy_image = torch.zeros_like(img_data[0]) 
                img_data.append(dummy_image)

            image_tensors = torch.stack(img_data)

            if len(image_tensors) == 0:
                raise ValueError(f"No images found in sample at index {index}")
            # Handle language/prompt
            prompt = sample["task"]
            if isinstance(prompt, (list, tuple)):
                prompt = prompt[0] if len(prompt) > 0 else ""
            if prompt is None:
                prompt = ""
            prompt_raw = sample["task"]
            # Process prompt with make_conversation
            prompt = make_conversation(prompt=prompt, conversation_type=0)
            # Use multi_image_get_item to process images and text
            results = multi_image_get_item(
                image_tensors=image_tensors,
                prompt=prompt,
                transform=self.transform,
                image_size=self.vla_image_resolution,
                text_tokenizer=self.tokenizer,
                num_image_token=self.num_image_token,
            )
            # Handle video for LAM (use first and last frame from original images)
            # Convert image_tensors to PIL for LAM transform
            
            
            video_pil_images = []

            for img_tensor in img_data_list[0]:
                if isinstance(img_tensor, torch.Tensor):
                    pil_img = to_pil_image(img_tensor)
                    video_pil_images.append(pil_img)
                else:
                    video_pil_images.append(img_tensor)         
               
            video = torch.stack([self.image_transform_lam(pil_image) for pil_image in video_pil_images], dim=0)

            # Build return dictionary
            ret = { 
                "input_ids": results["input_ids"],
                "labels": results["labels"],
                "attention_mask": results["attention_mask"],
                "position_ids": results["position_ids"],
                "pixel_values": results["pixel_values"],
                "image_flags": results["image_flags"],
                "videos": video,
                "prompts_raw": prompt_raw,
                "prompts": prompt,
                "dataset_name": self.dataset_name[dataset_idx],
            }
        elif self.stage == 'stage1-latent-pretrain':
            
            video_pil_images = []

            for img_tensor in img_data_list[0]:
                if isinstance(img_tensor, torch.Tensor):
                    pil_img = to_pil_image(img_tensor)
                    video_pil_images.append(pil_img)
                else:
                    video_pil_images.append(img_tensor)         
               
            video = torch.stack([self.image_transform_lam(pil_image) for pil_image in video_pil_images], dim=0)

            ret = { 
                "videos": video,
                "dataset_name": self.dataset_name[dataset_idx],
            }

        elif self.stage == 'stage3-vla-finetune':
            if self.use_augmentation:
           
                img_data = [
                    self.aug_transform(img) if random.random() < 0.5 else self.image_transform(img)
                    for img in pil_images
                ]

                # img_data = [
                #             self.aug_transform(pil_images[i]) if (random.random() < 0.5 and i>0) else self.image_transform(pil_images[i])
                #             for i in range(len(pil_images))
                #         ]
            else:
            
                img_data = [self.image_transform(img) for img in pil_images]
            image_tensors = torch.stack(img_data)

            if len(image_tensors) == 0:
                raise ValueError(f"No images found in sample at index {index}")
            # Handle language/prompt
            prompt = sample["task"]
            if isinstance(prompt, (list, tuple)):
                prompt = prompt[0] if len(prompt) > 0 else ""
            if prompt is None:
                prompt = ""
            
            prompt = self.action_space.preprocess_prompt(prompt)
            prompt_raw = prompt # sample["task"]
            # Process prompt with make_conversation
            prompt = make_conversation(prompt=prompt, conversation_type=0)

            # Use multi_image_get_item to process images and text
            results = multi_image_get_item(
                image_tensors=image_tensors,
                prompt=prompt,
                transform=self.transform,
                image_size=self.vla_image_resolution,
                text_tokenizer=self.tokenizer,
                num_image_token=self.num_image_token,
            )
            # Handle video for LAM (use first and last frame from original images)
            # Convert image_tensors to PIL for LAM transform
            
            video_pil_images = []

            for img_tensor in img_data_list[0]:
                if isinstance(img_tensor, torch.Tensor):
                    pil_img = to_pil_image(img_tensor)
                    video_pil_images.append(pil_img)
                else:
                    video_pil_images.append(img_tensor)         
               
            video = torch.stack([self.image_transform_lam(pil_image) for pil_image in video_pil_images], dim=0)

            
            arm_key = self.arm_keys[dataset_idx]
            

            # Check if state key exists in sample
            if self.state_key not in sample:
                raise KeyError(f"State key '{self.state_key}' not found in sample. Available keys: {list(sample.keys())}")
            
            
            

            # Use state key based on action_type
            state = sample[self.state_key]


            state = self.action_space.preprocess_state(state)


            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            else:
                state = state.float()

            # For multi-step state (ee6d_ear_step), keep full state for preprocess_action
            # but only use state[0] as model proprioception input
            if state.dim() > 1 and getattr(self.action_space, 'needs_multi_step_state', False):
                state_for_model = state[0]
            else:
                state_for_model = state

            state_padded, state_mask = self._pad_tensor(state_for_model, self.state_dim)

            # Check if action_type exists in sample
            if self.action_key not in sample:
                raise KeyError(f"Action type '{self.action_key}' not found in sample. Available keys: {list(sample.keys())}")
            
            

            # Use action_type to select action
            action = sample[self.action_key]
            action = self.action_space.preprocess_action(action,state)

            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.float32)
            else:
                action = action.float()


            action_padded, action_mask = self._pad_tensor(action, self.per_action_dim)

            action_padded,state_padded = self.action_space.norm(action=action_padded,state=state_padded )
            #print(state_padded[-1],state_padded.max(),state_padded.min(),action_padded.max(),action_padded.min())
            
            #print(torch.max(action_padded[:,:20],dim=0).values)
            ret = { 
                "input_ids": results["input_ids"],
                "labels": results["labels"],
                "attention_mask": results["attention_mask"],
                "position_ids": results["position_ids"],
                "pixel_values": results["pixel_values"],
                "image_flags": results["image_flags"],
                "videos": video,
                "prompts": prompt,
                "prompts_raw": prompt_raw,
                "dataset_name": self.dataset_name[dataset_idx],
                "actions": action_padded.to(dtype=torch.bfloat16),
                "action_mask": action_mask,
                "states": state_padded.to(dtype=torch.bfloat16),
                "state_mask": state_mask,
            }
        return ret

class LargeScaleWeightedRandomSampler(Sampler):
    def __init__(
        self, 
        weights: Union[torch.Tensor, list, np.ndarray], 
        num_samples: int, 
        num_samples_part: int,
        replacement: bool = True, 
        max_block: int = 2**24 - 1
    ):
        if isinstance(weights, list):
            weights = torch.tensor(weights)
        elif isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights)
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement
        self.max_block = max_block
        self.num_samples_part = num_samples_part

    def __iter__(self):
        return iter(self._sample_indices().tolist())

    def _sample_indices(self) -> torch.Tensor:
        weights = self.weights
        total_weight = weights.sum()
        indices = []
        n = len(weights)
        num_blocks = (n + self.max_block - 1) // self.max_block

        for i in range(num_blocks):
            start = i * self.max_block
            end = min((i + 1) * self.max_block, n)
            block_weights = weights[start:end].float()
            block_weight_sum = block_weights.sum()

            if block_weight_sum == 0:
                continue

            block_prob = block_weight_sum / total_weight
            block_sample_count = int(round(self.num_samples * block_prob.item()))
            sampled = torch.multinomial(block_weights, block_sample_count, self.replacement)
            indices.append(sampled + start)
        #print(self.num_samples,len(torch.cat(indices)))
        sample = min(self.num_samples,len(torch.cat(indices)))
        return torch.cat(indices)[:sample][torch.randperm(sample)[:self.num_samples_part]]  # truncate in case of rounding error

    def __len__(self):
        return self.num_samples_part



 
def get_window_size_for_dataset(dataset_name: str) -> int:

    datasets_with_lower_frequency = ['fractal20220817_data', 'toto', 'berkeley_autolab_ur5',#'bridge_oxe' 
        'nyu_franka_play_dataset_lerobot', 
        'ucsd_kitchen_dataset_lerobot', 
        'dlr_edan_shared_control_lerobot', 'dobbe']

        # From 15Hz to 30 Hz control frequency
    datasets_with_higher_frequency = ['utaustin_mutex', 
        'iamlab_cmu_pickup_insert_lerobot', 
        'austin_sirius_dataset_lerobot', 
        'austin_sailor_dataset_lerobot', 
        'toto', 'viola', 'droid']
    window_size = 10
    
    if 'task' == dataset_name[:4]: # AgiBot
        window_size = 30
    
    elif 'robomind' in dataset_name.lower():
        window_size = 25

    elif 'robocoin' in dataset_name.lower():
        window_size = 25

    elif 'ego4d' in dataset_name:
        window_size = 5

    elif dataset_name in datasets_with_lower_frequency:
        window_size = 4
    
    elif dataset_name in datasets_with_higher_frequency:
        window_size = 18
    elif 'libero' in dataset_name :
        window_size = 18

    elif 'aloha-agilex' in dataset_name : # robotwin
        window_size = 18
    #print(dataset_name,window_size)
    return window_size


def create_lerobot_dataset(
    data_cfg: dict,
    tokenizer = None,
    stage: str = "stage1-latent-pretrain",
    horizon: int = 30,
    image_horizon: int = 2,
    #num_image_token: int = 256,
    shuffle: bool = False,
    is_log: bool = False,
):
    """
    Create a lerobot MixtureDataset from config.
    
    Args:
        tokenizer: Tokenizer for text processing
        stage: Training stage
        horizon: Number of action steps to predict
        image_horizon: Number of images steps as input

    Returns:
        training_dataset: LeRobotDatasetAdapter instance
        training_ds_weights: Sample weights array
    """

    data_info = data_cfg.get("data_info", "")
    use_augmentation=data_cfg["use_augmentation"]
    data_mix = data_cfg["data_mix"]
    max_views = data_cfg["max_views"]
    vla_image_resolution = data_cfg.get("vla_image_resolution", 448)

    mixture_spec = OXE_NAMED_MIXTURES[data_mix]
    if stage == 'stage1-latent-pretrain' :
        num_image_token = None
    else :
        num_image_token = int(
            (vla_image_resolution // data_cfg["patch_size"]) ** 2 * (data_cfg["downsample_ratio"]**2)
        )

    dir_dict = []
    weights = []
    
    if  is_log:
        print(f"Using mixture '{data_mix}' from OXE_NAMED_MIXTURES with {len(mixture_spec)} datasets")
    
    for ds_name, weight in mixture_spec:

        local_path = data_info[ds_name]["path"]

        
        dir_dict.append([ds_name, local_path])
        weights.append(weight)
                
    
    if is_log:
        print(f"Found {len(dir_dict)} datasets:")
        for ds_name, local_path in dir_dict:
            print(f"  - {ds_name}: {local_path}")
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("Total weight is 0, cannot create dataset")
    datasets_weights_raw = {ds_name: w for (ds_name, _), w in zip(dir_dict, weights)}
    weights = [w / total_weight for w in weights]
    datasets_weights = {ds_name: w for (ds_name, _), w in zip(dir_dict, weights)}

    # Create training datasets
    all_training_datasets = []
    all_training_datasets_name = []
    all_training_datasets_meta = []
    arm2stats_list = []
    arm2stats_dict = {}

    #action_type = data_cfg["action_type"] # "action"
    action_space =build_action_space(data_cfg.get("action_space", "ee6d"), norm_path=data_cfg.get("norm_path", None),delta_action=data_cfg.get("delta_action", False))
    
    for i, (ds_name, local_path) in enumerate(dir_dict):
        if not os.path.exists(local_path):
            print(f"Dataset path {local_path} does not exist, skipping...")
            continue

        if 'AgiBotWorld' in local_path : # Use the meta info without fisheye to accelerate data loading
            local_path_meta = local_path.replace('AgiBotWorld-Alpha-lerobot','AgiBotWorld-Alpha-meta')
            local_path_meta = local_path_meta.replace('AgiBotWorld-Alpha-LeRobot','AgiBotWorld-Alpha-meta')
        else :
            local_path_meta = local_path

        ds_meta = LeRobotDatasetMetadata(root=local_path_meta, repo_id=os.path.basename(local_path))
        if stage in ['stage1-latent-pretrain','stage2-vla-pretrain']:
            delta_timestamps = {
                data_info[ds_name]["view_map"]["image_1"]: [t*get_window_size_for_dataset(ds_name) / ds_meta.fps for t in range(image_horizon)]
            }

        elif stage == 'stage3-vla-finetune':
            delta_timestamps = {
                action_space.get_action_key(): [t / ds_meta.fps for t in range(horizon)],
                data_info[ds_name]["view_map"]["image_1"]: [t*get_window_size_for_dataset(ds_name) / ds_meta.fps for t in range(image_horizon)],
            }
            # Load multi-step states if action space requires it
            if getattr(action_space, 'needs_multi_step_state', False):
                delta_timestamps[action_space.get_state_key()] = [t / ds_meta.fps for t in range(horizon)]

        training_dataset = LeRobotDataset(
            local_path,
            delta_timestamps=delta_timestamps,
            video_backend="torchcodec",
            tolerance_s=0.001,
            #revision=V21
        )
            
        meta_info = {
            "num_episodes": training_dataset.num_episodes, 
            "num_frames": training_dataset.num_frames,
            "fps": training_dataset.fps,
            "num_indices": data_cfg.get("num_indices"),
            "robot_type": ds_meta.info["robot_type"],
        }

        all_training_datasets_name.append(ds_name)
        all_training_datasets_meta.append(meta_info)
        all_training_datasets.append(training_dataset)
        if ds_meta.info["robot_type"] not in arm2stats_dict.keys():
            arm2stats_dict[ds_meta.info["robot_type"]] = []
        arm2stats_dict[ds_meta.info["robot_type"]].append(ds_meta.stats)

    arm2stats_dict_merge = None

    if len(all_training_datasets) == 0:
        raise ValueError("No valid datasets found!")

    mixture_dataset = MixtureDataset(
        all_training_datasets, 
        all_training_datasets_name,
        all_training_datasets_meta,
        datasets_weights,
        datasets_weights_raw,
        stage=stage,
        shuffle=shuffle,
    )
    training_ds_weights = mixture_dataset.sample_weights
    if is_log:
        logging.info("\n" + mixture_dataset.to_log_string())

    image_transform_lam = T.Compose([
        T.Resize((data_cfg.get("resolution", 224), data_cfg.get("resolution", 224))),
        T.ToTensor()
    ])
    image_transform = T.Compose([
        T.Resize((data_cfg.get("vla_image_resolution", 224), data_cfg.get("vla_image_resolution", 224)), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor()
    ])

    if use_augmentation : 
        # aug_transform = T.Compose([
        #     T.Resize((data_cfg.get("vla_image_resolution", 224), data_cfg.get("vla_image_resolution", 224)), interpolation=InterpolationMode.BICUBIC),
        #     T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
        #     T.ToTensor()
        # ])
        aug_transform = T.Compose([
            # Resize first, exactly like image_transform: the augmented and the
            # plain path are mixed per-view inside one sample, so they must agree
            # on the output size or torch.stack over views fails.
            T.Resize((data_cfg.get("vla_image_resolution", 224), data_cfg.get("vla_image_resolution", 224)), interpolation=InterpolationMode.BICUBIC),
            T.RandomAffine(degrees=0,translate=(0.05, 0.05), interpolation=InterpolationMode.BICUBIC),
            T.RandomRotation(degrees=(-10, 10), interpolation=InterpolationMode.BICUBIC) ,
            T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
            T.ToTensor()
        ])

    else :
        aug_transform = None
    
    training_dataset = LeRobotDatasetAdapter(
        mixture_dataset,
        tokenizer=tokenizer,
        image_transform=image_transform,
        aug_transform=aug_transform,
        image_transform_lam=image_transform_lam,
        vla_image_resolution=vla_image_resolution,
        num_image_token=num_image_token,
        data_info=data_info,
        use_augmentation=use_augmentation,
        stage=data_cfg["stage"],
        max_views=max_views,
        arm2stats_dict=arm2stats_dict_merge,
        state_dim=data_cfg.get("state_dim", 24),
        per_action_dim=data_cfg.get("per_action_dim", 24),
        action_space_type=data_cfg.get("action_space", "ee6d"),
        delta_action=data_cfg.get("delta_action", False),
        norm_path=data_cfg.get("norm_path", None),
    )

    return training_dataset, training_ds_weights

