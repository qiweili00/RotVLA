#!/usr/bin/env python3
"""RotVLA HTTP inference server.

Serves an ``/act`` endpoint for the LIBERO and RoboTwin evaluation clients under
``experiment/``. Dispatches on the ``action_space`` recorded in the checkpoint config.
"""

import sys
import os
import json
import argparse
import logging
import socket
from flask import Flask, request, jsonify
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
import json_numpy
import robosuite.utils.transform_utils as Trans
from typing import List, Union, Dict, Any, Sequence

# On many-core machines PyTorch defaults to one CPU thread per core (here 72).
# The image preprocessing (ToTensor/Normalize on tiny 224x224 tensors) is CPU-bound,
# and thread scheduling overhead across 72 threads makes it ~0.6s/request instead of
# ~1ms. Cap CPU threads so preprocessing is fast; GPU inference is unaffected.
torch.set_num_threads(4)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from model.rotvla import RotVLA
import time 

app = Flask(__name__)

# Global variables for model and normalizer
global model, normalizer, config_data
model = None
normalizer = None
num_view = 3
config_data = None



def load_model(ckpt_dir, time_step=20):
    config = json.load(open(os.path.join(ckpt_dir, "config.json")))

    config["finetune_vlm"] = False
    config["finetune_action_head"] = False
    config["num_inference_timesteps"] = time_step
    
    model = RotVLA(config,train=False).eval()
        
    ckpt_path = os.path.join(ckpt_dir, "mp_rank_00_model_states.pt")
    checkpoint = torch.load(ckpt_path, map_location="cpu",weights_only=False)
    #model.load_state_dict(checkpoint["module"], strict=False)
    #model.load_state_dict(checkpoint["module"], strict=True)
    ckpt = checkpoint["module"]
    ckpt_load = {}
    for key in ckpt.keys():
        if not ('action_latent' in key):
            ckpt_load[key] = ckpt[key]
    
    model.load_state_dict(ckpt_load, strict=False)
    model = model.to("cuda")
    
    return model


def decode_image_from_array(img_array, image_size):
    """Convert numpy array to tensor"""
    if isinstance(img_array, list):
        img_array = np.array(img_array, dtype=np.uint8)
    img = cv2.resize(img_array, (image_size, image_size))
    pil = Image.fromarray(img)
    return transforms.ToTensor()(pil).to("cuda")


def infer_from_data(data, model, normalizer, num_view, config_data):
    """Inference function for CALVIN data format"""
    device = "cuda"
    
    # Decode images
    image0 = json_numpy.loads(data["image0"])
    image1 = json_numpy.loads(data["image1"])
    if "image2" in data:
        image2 = json_numpy.loads(data["image2"])
    else:
        image2 = np.zeros((config_data["vla_image_resolution"], config_data["vla_image_resolution"], 3), dtype=np.uint8)    
        # Create dummy third image if needed (use a standard size, will be resized anyway)
 
    images = [
        decode_image_from_array(image0, config_data["vla_image_resolution"]),
        decode_image_from_array(image1, config_data["vla_image_resolution"]),
        decode_image_from_array(image2, config_data["vla_image_resolution"])
    ]
    
    images = images[:num_view]
    assert len(images) == num_view, f"Must provide exactly {num_view} images."
    
    # Process state (proprio)
    proprio = json_numpy.loads(data["proprio"])
    if isinstance(proprio, list):
        proprio = np.array(proprio)
    
    if len(proprio) >= 10:
        state = proprio
    else:
        state = np.pad(proprio, (0, 10 - len(proprio)))
    
    # Convert to tensor and pad to 24
    state = torch.tensor(state, dtype=torch.float32, device=device)
    if state.ndim == 1:
        state = state.unsqueeze(0)

    if state.shape[1] < 24:
        state = torch.cat([state, torch.zeros((1, 24 - state.shape[1]), device=device)], dim=1)

    action_space = config_data.get("action_space", 'eepose')
    if action_space == 'ee6d':
        norm_state = state.to(dtype=torch.float32)

    elif action_space == 'ee6d_robotwin':
        norm_state = state.to(dtype=torch.float32)
    elif action_space == 'ee6d_libero':
        norm_state = state.to(dtype=torch.float32)

    else:
        raise ValueError(f"Unknown action_space: {action_space}")
    

    # Prepare prompt and masks
    prompt = data["language_instruction"]
    
    
    # Action mask: based on action_space
    if action_space == 'eepose':
        action_len = 7
    elif action_space == 'ee6d':
        action_len = 10
    elif action_space == 'ee6d_libero':
        action_len = 10
    elif action_space == 'ee6d_robotwin':
        action_len = 20
    else:
        action_len = 7
    
    action_mask = torch.tensor([[1] * action_len + [0] * (24 - action_len)], dtype=torch.int32, device=device)
    
    # Run inference
    
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            action = model.run_inference(
                images=images,
                prompt=prompt,
                state_input=norm_state,
                action_mask=action_mask
            )
            
            action = action.reshape(1, -1, 24)
            
            action = model.action_space.postprocess(action,norm_state)[0]

    
    # Convert to numpy and return as list
    action_np = action.to(dtype=torch.float32).cpu().numpy()
    

    action_list = []
    for i in range(action_np.shape[0]):
        action_list.append(action_np[i].tolist())
    
    return action_list


def warmup(model, normalizer, num_view, config_data, n=3):
    """Run a few dummy inferences to trigger cudnn autotune, lazy init and
    allocator warmup so the first real request isn't ~3x slower than steady state."""
    res = config_data["vla_image_resolution"]
    dummy_img = np.zeros((res, res, 3), dtype=np.uint8)
    data = {
        "language_instruction": "warmup",
        "proprio": json_numpy.dumps(np.zeros(10, dtype=np.float32)),
        "image0": json_numpy.dumps(dummy_img),
        "image1": json_numpy.dumps(dummy_img),
        "image2": json_numpy.dumps(dummy_img),
    }
    print(f"Warming up ({n} iters)...")
    for i in range(n):
        t0 = time.time()
        infer_from_data(data, model, normalizer, num_view, config_data)
        print(f"  warmup {i}: {time.time()-t0:.3f}s")
    torch.cuda.synchronize()
    print("Warmup done.")


@app.route('/act', methods=['POST'])
def act():
    """Handle action prediction requests from CALVIN client"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ["language_instruction", "proprio", "image0", "image1"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Run inference
        action_list = infer_from_data(data, model, normalizer, num_view, config_data)
        return jsonify({"action": action_list})
    
    except Exception as e:
        logging.error(f"Error in /act endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    #global model, normalizer, config_data
    
    parser = argparse.ArgumentParser(description="Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--time_step", type=int, default=20, help="Number of inference timesteps")
    args = parser.parse_args()
    
    # Load config
    config_path = os.path.join(args.ckpt_dir, "config.json")
    import yaml
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    

    
    print("Loading model...")
    model = load_model(args.ckpt_dir, args.time_step)
    #print(f"Model loaded. num_view={num_view}")

    warmup(model, normalizer, num_view, config_data)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    
    print(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=False)
