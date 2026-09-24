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
from traceback import print_tb
from typing import Iterable, Tuple, Dict, Type
import torch
import torch.nn as nn
import json
import numpy as np
import robosuite.utils.transform_utils as Trans
# =============================================================================
# Registry
# =============================================================================

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



ACTION_REGISTRY: Dict[str, Type["BaseActionSpace"]] = {}


def register_action(name: str):
    """Decorator for registering a new action space."""
    def _wrap(cls):
        key = name.lower()
        if key in ACTION_REGISTRY:
            raise KeyError(f"ActionSpace '{key}' already registered -> {ACTION_REGISTRY[key]}")
        ACTION_REGISTRY[key] = cls
        cls.name = key
        return cls
    return _wrap


def build_action_space(name: str, **kwargs) -> "BaseActionSpace":
    """Instantiate a registered action space by name."""
    key = name.lower()
    if key not in ACTION_REGISTRY:
        raise KeyError(f"Unknown action space '{name}'. Available: {list(ACTION_REGISTRY.keys())}")
    return ACTION_REGISTRY[key](**kwargs)


# =============================================================================
# Base class
# =============================================================================
class BaseActionSpace(nn.Module):
    """
    Abstract base class for all action-space definitions.

    Each subclass defines:
      - `dim_action`: dimension of the action vector.
      - `gripper_idx`: indices of gripper channels.
      - `compute_loss(pred, target)`: supervised loss for this space.
      - `preprocess(proprio, action, mode)`: pre-step modifications.
      - `postprocess(action)`: post-step corrections (e.g. apply sigmoid).
    """

    name: str = "base"
    dim_action: int = 0
    gripper_idx: Tuple[int, ...] = ()

    def __init__(self,norm_path):
        super().__init__()
        self.norm_path = norm_path
        if norm_path is not None :
            with open(norm_path, "r") as f:
                norm_file = json.load(f)
            self.state_q01 =np.array( norm_file["norm_stats"]["state"]["q01"])
            self.state_q99 =np.array( norm_file["norm_stats"]["state"]["q99"])
            self.actions_q01 =np.array( norm_file["norm_stats"]["actions"]["q01"])
            self.actions_q99 =np.array( norm_file["norm_stats"]["actions"]["q99"])

            self.state_len = len(self.state_q01)
            self.actions_len = len(self.actions_q01)
    # ---------------------------------------------------------------------
    # Core supervised loss
    # ---------------------------------------------------------------------
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Alias for compute_loss."""
        return self.compute_loss(pred, target)

    # ---------------------------------------------------------------------
    # Space-level hooks
    # ---------------------------------------------------------------------
    def preprocess(
        self,
        proprio: torch.Tensor,
        action: torch.Tensor,
        mode: str = "train",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Default: return unchanged."""
        return proprio, action

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Default: return unchanged."""
        return action
    
    def norm(self,action=None,state=None): 
        
        if self.norm_path is not None :
            if state is not None :
                state[..., :self.state_len] = (state[..., :self.state_len] - self.state_q01) / (self.state_q99 - self.state_q01 + 1e-6) * 2.0 - 1.0
            if action is not None :
                action[..., :self.actions_len] = (action[..., :self.actions_len] - self.actions_q01) / (self.actions_q99 - self.actions_q01 + 1e-6) * 2.0 - 1.0
        return action,state

    def denorm(self,action=None,state=None):

        if self.norm_path is not None :
            if state is not None :
                state[..., :self.state_len] = (state[..., :self.state_len] + 1.0) / 2.0 * (self.state_q99 - self.state_q01 + 1e-6) + self.state_q01
            if action is not None :
                action[..., :self.actions_len] = (action[..., :self.actions_len] + 1.0) / 2.0 * (self.actions_q99 - self.actions_q01 + 1e-6) + self.actions_q01
        return action,state 
# =============================================================================
# Utilities
# =============================================================================
def _ensure_indices_valid(D: int, idx: Iterable[int], name: str) -> None:
    bad = [i for i in idx if i < 0 or i >= D]
    if bad:
        raise IndexError(f"{name} contains out-of-range indices {bad} for action dim D={D}")


# =============================================================================
# Implementations
# =============================================================================

@register_action("ee6d")
@register_action("ee6d_libero")
class EE6DActionSpace(BaseActionSpace):
    """End-effector layout with xyz, 6D rotation, and gripper channels."""

    dim_action = 24
    gripper_idx = [9]
    GRIPPER_SCALE = 1.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0
    SCALE = 1.0
    POS_IDX_1 = [0, 1, 2]
    #POS_IDX_2 = (10, 11, 12)
    ROT_IDX_1 = [3, 4, 5, 6, 7, 8]
    #ROT_IDX_2 = (13, 14, 15, 16, 17, 18)
    
    max_action_dim = 10

    delta_mask =  [True] * 3 + [False] * 7 

    def __init__(self,norm_path,delta_action):
        super().__init__(norm_path)
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.delta_action = delta_action

    def get_action_key(self):
        return 'action_abs'

    def get_state_key(self):
        return 'observation.states.robot_ee6d_state'

    def preprocess_state(self, state):
        return state 
    
    def preprocess_action(self,action,state=None):
        action_abs = [np.concatenate([action_abs_i[:3], AxisAngle_to_Rotate6D(action_abs_i[3:6]).tolist(), action_abs_i[6:]]) for action_abs_i in action]
        
        if self.delta_action:
            dims = len(self.delta_mask)
            action_abs[..., :dims] -= np.expand_dims(np.where(self.delta_mask, state[..., :dims], 0), axis=-2)
        return action_abs



    def preprocess_prompt(self, prompt) :
        return prompt
        

    def compute_loss(self, pred, target):
        assert pred.shape == target.shape, "pred/target shapes must match"
        B, T, D = pred.shape
        #print(pred.shape)
        _ensure_indices_valid(D, self.gripper_idx, "gripper_idx")

        # Gripper BCE
        g_losses = [self.bce(pred[:, :, gi], target[:, :, gi]) for gi in self.gripper_idx]
        gripper_loss = sum(g_losses) / len(self.gripper_idx) * self.GRIPPER_SCALE

        # XYZ position
        pos_loss = (
            self.mse(pred[:, :, self.POS_IDX_1], target[:, :, self.POS_IDX_1]) 
        ) * self.XYZ_SCALE

        # Rotation 6D
        rot_loss = (
            self.mse(pred[:, :, self.ROT_IDX_1], target[:, :, self.ROT_IDX_1])
        ) * self.ROT_SCALE

        action_loss = (pos_loss + rot_loss + gripper_loss) #/ (self.XYZ_SCALE + self.ROT_SCALE + self.GRIPPER_SCALE)

        return {
            "action_loss": action_loss,
            "position_loss": pos_loss,#/ (self.XYZ_SCALE + self.ROT_SCALE + self.GRIPPER_SCALE),
            "rotate6D_loss": rot_loss,#/ (self.XYZ_SCALE + self.ROT_SCALE + self.GRIPPER_SCALE),
            "gripper_loss": gripper_loss,#/ (self.XYZ_SCALE + self.ROT_SCALE + self.GRIPPER_SCALE),
        }
    
    def compute_latent_action_loss(self, pred, target):
        assert pred.shape == target.shape, "pred/target shapes must match"
        action_loss = self.mse(pred, target)  * self.SCALE
        return {
            "action_loss": action_loss,
        }
    
    def preprocess(self, proprio, action, mode="train"):
        """Zero-out gripper channels in proprio/action."""
        proprio_m = proprio.clone()
        action_m = action.clone()
        proprio_m[..., self.gripper_idx] = 0.0
        action_m[..., self.gripper_idx] = 0.0
        # if max_state_dim > 0:
        #     proprio_m[..., max_state_dim:] = 0.0
        # else :
        #     proprio_m[..., self.max_action_dim:] = 0.0
        action_m[..., self.max_action_dim:] = 0.0
        return proprio_m, action_m

    def preprocess_gt(self, action, mode="train"):
        """Zero-out gripper channels in proprio/action."""
        EPSILON = 1e-6

        g = action[..., self.gripper_idx]

        action[..., self.gripper_idx] = torch.where(
            g > 0.5 - EPSILON,
            torch.ones_like(g),
            torch.zeros_like(g)
        )
        return action

    def postprocess(self, action: torch.Tensor,state = None) -> torch.Tensor:
        """Apply sigmoid to gripper logits."""
        if self.delta_action :
            dims = action.shape[-1]
            action += np.expand_dims(np.where(self.delta_mask, state[..., :dims], 0), axis=-2)
        if action.size(-1) > max(self.gripper_idx):
            action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        return action



@register_action("ee6d_robotwin")
class EE6D_RobotwinActionSpace(BaseActionSpace):
    """End-effector layout with xyz, 6D rotation, and gripper channels."""

    dim_action = 24
    gripper_idx = [9,19]
    GRIPPER_SCALE = 1.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0
    SCALE = 1.0
    POS_IDX_1 = [0, 1, 2]
    POS_IDX_2 = (10, 11, 12)
    ROT_IDX_1 = [3, 4, 5, 6, 7, 8]
    ROT_IDX_2 = (13, 14, 15, 16, 17, 18)
    max_action_dim = 20 

    delta_mask =  [True] * 3 + [False] * 7 +  [True] * 3 + [False] * 7

    def __init__(self,norm_path,delta_action):
        super().__init__(norm_path)
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.delta_action = delta_action

    def get_action_key(self):
        return 'action_eef6d'

    def get_state_key(self):
        return 'observation.state_eef6d'

    def preprocess_state(self, state):
        return state 
    
    def preprocess_action(self,action,state=None):
        if self.delta_action:
            dims = len(self.delta_mask)
            action[..., :dims] -= np.expand_dims(np.where(self.delta_mask, state[..., :dims], 0), axis=-2)
        return action

    def preprocess_prompt(self, prompt) :
        return prompt

    def compute_loss(self, pred, target):
        assert pred.shape == target.shape, "pred/target shapes must match"
        B, T, D = pred.shape
        #print(pred.shape)
        _ensure_indices_valid(D, self.gripper_idx, "gripper_idx")

        # Gripper BCE
        #print(torch.unique( target[:, :, self.gripper_idx]))
        g_losses = [self.bce(pred[:, :, gi], target[:, :, gi]) for gi in self.gripper_idx]
        gripper_loss = sum(g_losses) / len(self.gripper_idx) * self.GRIPPER_SCALE

        # XYZ position
        pos_loss = (
            self.mse(pred[:, :, self.POS_IDX_1], target[:, :, self.POS_IDX_1]) +
            self.mse(pred[:, :, self.POS_IDX_2], target[:, :, self.POS_IDX_2])
        ) * self.XYZ_SCALE

        # Rotation 6D
        rot_loss = (
            self.mse(pred[:, :, self.ROT_IDX_1], target[:, :, self.ROT_IDX_1]) +
            self.mse(pred[:, :, self.ROT_IDX_2], target[:, :, self.ROT_IDX_2])
        ) * self.ROT_SCALE

        action_loss = (pos_loss + rot_loss + gripper_loss) #/ (self.XYZ_SCALE + self.ROT_SCALE + self.GRIPPER_SCALE)

        return {
            "action_loss": action_loss,
            "position_loss": pos_loss,#/ (self.XYZ_SCALE + self.ROT_SCALE + self.GRIPPER_SCALE),
            "rotate6D_loss": rot_loss,#/ (self.XYZ_SCALE + self.ROT_SCALE + self.GRIPPER_SCALE),
            "gripper_loss": gripper_loss,#/ (self.XYZ_SCALE + self.ROT_SCALE + self.GRIPPER_SCALE),
        }
    def compute_latent_action_loss(self, pred, target):
        assert pred.shape == target.shape, "pred/target shapes must match"
        action_loss = self.mse(pred, target)  * self.SCALE
        return {
            "action_loss": action_loss,
        }
    def preprocess(self, proprio, action, mode="train"):
        """Zero-out gripper channels in proprio/action."""
        proprio_m = proprio.clone()
        action_m = action.clone()
        proprio_m[..., self.gripper_idx] = 0.0
        action_m[..., self.gripper_idx] = 0.0
        # if max_state_dim > 0:
        #     proprio_m[..., max_state_dim:] = 0.0
        # else :
        #     proprio_m[..., self.max_action_dim:] = 0.0
        action_m[..., self.max_action_dim:] = 0.0
        return proprio_m, action_m

    def preprocess_gt(self, action, mode="train"):
        """Zero-out gripper channels in proprio/action."""
        EPSILON = 5e-2

        g = action[..., self.gripper_idx]

        action[..., self.gripper_idx] = torch.where(
            g > 0.5 - EPSILON,
            torch.ones_like(g),
            torch.zeros_like(g)
        )
        #print(action[..., self.gripper_idx].shape,torch.sum(action[..., self.gripper_idx]))
        return action

    def postprocess(self, action: torch.Tensor,state = None) -> torch.Tensor:
        """Apply sigmoid to gripper logits."""
        if self.delta_action :
            dims = action.shape[-1]
            action += np.expand_dims(np.where(self.delta_mask, state[..., :dims], 0), axis=-2)
        if action.size(-1) > max(self.gripper_idx):
            action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        return action


@register_action("eequat_ear")
class EEQuat_EarActionSpace(BaseActionSpace):
    """End-effector layout with xyz, 6D rotation, and gripper channels."""

    dim_action = 24
    gripper_idx = [7,15]
    GRIPPER_SCALE = 1.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0
    SCALE = 1.0
    POS_IDX_1 = [0, 1, 2]
    POS_IDX_2 = (8, 9, 10)
    ROT_IDX_1 = [3, 4, 5, 6,]
    ROT_IDX_2 = (11, 12, 13, 14)
    max_action_dim = 16 

    delta_mask =  [True] * 3 + [False] * 5 +  [True] * 3 + [False] * 5

    def __init__(self,norm_path,delta_action):
        super().__init__(norm_path)
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.delta_action = delta_action

    def get_action_key(self):
        return 'action_quat'

    def get_state_key(self):
        return 'observation.state_quat'

    def preprocess_state(self, state):
        return state 
    

    def preprocess_action(self,action,state=None):
        if self.delta_action:
            dims = len(self.delta_mask)
            action[..., :dims] -= np.expand_dims(np.where(self.delta_mask, state[..., :dims], 0), axis=-2)
        return action
    def preprocess_prompt(self, prompt) :
        return 'Pack earphone'

    def compute_loss(self, pred, target):
        assert pred.shape == target.shape, "pred/target shapes must match"
        B, T, D = pred.shape
        #print(pred.shape)
        _ensure_indices_valid(D, self.gripper_idx, "gripper_idx")

        # Gripper BCE
        #print(torch.unique( target[:, :, self.gripper_idx]))
        g_losses = [self.bce(pred[:, :, gi], target[:, :, gi]) for gi in self.gripper_idx]
        gripper_loss = sum(g_losses) / len(self.gripper_idx) * self.GRIPPER_SCALE

        # XYZ position
        pos_loss = (
            self.mse(pred[:, :, self.POS_IDX_1], target[:, :, self.POS_IDX_1]) +
            self.mse(pred[:, :, self.POS_IDX_2], target[:, :, self.POS_IDX_2])
        ) * self.XYZ_SCALE

        # Rotation 6D
        rot_loss = (
            self.mse(pred[:, :, self.ROT_IDX_1], target[:, :, self.ROT_IDX_1]) +
            self.mse(pred[:, :, self.ROT_IDX_2], target[:, :, self.ROT_IDX_2])
        ) * self.ROT_SCALE

        action_loss = (pos_loss + rot_loss + gripper_loss) #/ (self.XYZ_SCALE + self.ROT_SCALE + self.GRIPPER_SCALE)

        return {
            "action_loss": action_loss,
            "position_loss": pos_loss,#/ (self.XYZ_SCALE + self.ROT_SCALE + self.GRIPPER_SCALE),
            "rotate6D_loss": rot_loss,#/ (self.XYZ_SCALE + self.ROT_SCALE + self.GRIPPER_SCALE),
            "gripper_loss": gripper_loss,#/ (self.XYZ_SCALE + self.ROT_SCALE + self.GRIPPER_SCALE),
        }
    def compute_latent_action_loss(self, pred, target):
        assert pred.shape == target.shape, "pred/target shapes must match"
        action_loss = self.mse(pred, target)  * self.SCALE
        return {
            "action_loss": action_loss,
        }
    def preprocess(self, proprio, action, mode="train"):
        """Zero-out gripper channels in proprio/action."""
        proprio_m = proprio.clone()
        action_m = action.clone()
        proprio_m[..., self.gripper_idx] = 0.0
        action_m[..., self.gripper_idx] = 0.0
        # if max_state_dim > 0:
        #     proprio_m[..., max_state_dim:] = 0.0
        # else :
        #     proprio_m[..., self.max_action_dim:] = 0.0
        action_m[..., self.max_action_dim:] = 0.0
        return proprio_m, action_m

    def preprocess_gt(self, action, mode="train"):
        """Zero-out gripper channels in proprio/action."""
        EPSILON = 5e-2

        g = action[..., self.gripper_idx]

        action[..., self.gripper_idx] = torch.where(
            g > -1.5 - EPSILON,
            torch.ones_like(g),
            torch.zeros_like(g)
        )
        #print(action[..., self.gripper_idx].shape,torch.sum(action[..., self.gripper_idx]))
        return action

    def postprocess(self, action: torch.Tensor,state = None) -> torch.Tensor:
        """Apply sigmoid to gripper logits."""
        #print('enter postprocess')
        if self.delta_action :
            dims = action.shape[-1]
            action += np.expand_dims(np.where(self.delta_mask, state[..., :dims], 0), axis=-2)
        if action.size(-1) > max(self.gripper_idx):
            action[..., self.gripper_idx] = (-3) * (1 - torch.sigmoid(action[..., self.gripper_idx]))
        return action


@register_action("ee6d_ear_step")
class EE6D_EarStepActionSpace(BaseActionSpace):
    """End-effector layout with xyz, 6D rotation, and gripper channels.

    Unlike ee6d_ear which computes delta as action[t] - state[0] (all steps
    relative to the first state), this space computes per-step deltas:
      delta[t] = action[t] - state[t]   for each t
    Requires multi-step state loading (needs_multi_step_state = True).
    """

    dim_action = 24
    gripper_idx = [9, 19]
    GRIPPER_SCALE = 1.0
    XYZ_SCALE = 50.0
    ROT_SCALE = 10.0
    SCALE = 1.0
    POS_IDX_1 = [0, 1, 2]
    POS_IDX_2 = (10, 11, 12)
    ROT_IDX_1 = [3, 4, 5, 6, 7, 8]
    ROT_IDX_2 = (13, 14, 15, 16, 17, 18)
    max_action_dim = 20

    delta_mask = [True] * 3 + [False] * 7 + [True] * 3 + [False] * 7
    needs_multi_step_state = True

    def __init__(self, norm_path, delta_action):
        super().__init__(norm_path)
        self.mse = nn.MSELoss()
        self.delta_action = delta_action

    def get_action_key(self):
        return 'action'

    def get_state_key(self):
        return 'observation.state'

    def preprocess_state(self, state):
        return state[..., [0, 1, 2, 3, 6, 9, 4, 7, 10, 12, 13, 14, 15, 16, 19, 22, 17, 20, 23, 25]]

    def preprocess_action(self, action, state=None):
        action = action[..., [0, 1, 2, 3, 6, 9, 4, 7, 10, 12, 13, 14, 15, 16, 19, 22, 17, 20, 23, 25]]
        if self.delta_action:
            dims = len(self.delta_mask)
            mask = np.array(self.delta_mask, dtype=np.float32)
            # Per-step delta: delta[t] = action[t] - state[t]
            # state shape: (horizon, dims) — each step has its own state
            if hasattr(state, 'numpy'):
                state = state.numpy()
            if hasattr(action, 'numpy'):
                action = action.numpy()
            action[..., :dims] -= mask * state[..., :dims]
        return action

    def preprocess_prompt(self, prompt):
        return 'Pack earphone'

    def compute_loss(self, pred, target):
        assert pred.shape == target.shape, "pred/target shapes must match"
        B, T, D = pred.shape
        _ensure_indices_valid(D, self.gripper_idx, "gripper_idx")

        gripper_loss = (
            self.mse(pred[:, :, self.gripper_idx], target[:, :, self.gripper_idx])
        ) * self.GRIPPER_SCALE

        pos_loss = (
            self.mse(pred[:, :, self.POS_IDX_1], target[:, :, self.POS_IDX_1]) +
            self.mse(pred[:, :, self.POS_IDX_2], target[:, :, self.POS_IDX_2])
        ) * self.XYZ_SCALE

        rot_loss = (
            self.mse(pred[:, :, self.ROT_IDX_1], target[:, :, self.ROT_IDX_1]) +
            self.mse(pred[:, :, self.ROT_IDX_2], target[:, :, self.ROT_IDX_2])
        ) * self.ROT_SCALE

        action_loss = pos_loss + rot_loss + gripper_loss

        return {
            "action_loss": action_loss,
            "position_loss": pos_loss,
            "rotate6D_loss": rot_loss,
            "gripper_loss": gripper_loss,
        }

    def compute_latent_action_loss(self, pred, target):
        assert pred.shape == target.shape, "pred/target shapes must match"
        action_loss = self.mse(pred, target) * self.SCALE
        return {
            "action_loss": action_loss,
        }

    def preprocess(self, proprio, action, mode="train"):
        """Zero-out padding dimensions beyond max_action_dim."""
        proprio_m = proprio.clone()
        action_m = action.clone()
        action_m[..., self.max_action_dim:] = 0.0
        return proprio_m, action_m

    def preprocess_gt(self, action, mode="train"):
        return action

    def postprocess(self, action: torch.Tensor, state=None) -> torch.Tensor:
        """Reconstruct absolute actions via cumulative sum of per-step deltas."""
        if self.delta_action:
            dims = len(self.delta_mask)
            mask = np.array(self.delta_mask, dtype=np.float32)
            # First step: add back initial state
            action[..., 0, :dims] += mask * state[..., :dims]
            # Subsequent steps: cumulative sum to recover absolute poses
            for t in range(1, action.shape[-2]):
                action[..., t, :dims] += mask * action[..., t - 1, :dims]
        return action


@register_action("ee6d_ear")
class EE6D_EarActionSpace(BaseActionSpace):
    """End-effector layout with xyz, 6D rotation, and gripper channels."""

    dim_action = 24
    gripper_idx = [9,19]
    GRIPPER_SCALE = 1.0
    XYZ_SCALE = 50.0
    ROT_SCALE = 10.0
    SCALE = 1.0
    POS_IDX_1 = [0, 1, 2]
    POS_IDX_2 = (10, 11, 12)
    ROT_IDX_1 = [3, 4, 5, 6, 7, 8]
    ROT_IDX_2 = (13, 14, 15, 16, 17, 18)
    max_action_dim = 20 

    delta_mask =  [True] * 3 + [False] * 7 +  [True] * 3 + [False] * 7

    def __init__(self,norm_path,delta_action):
        super().__init__(norm_path)
        self.mse = nn.MSELoss()
        #self.bce = nn.BCEWithLogitsLoss()
        self.delta_action = delta_action
    def get_action_key(self):
        return 'action'

    def get_state_key(self):
        return 'observation.state'

    def preprocess_state(self, state):
        return state[..., [0,1,2,3,6,9,4,7,10,12,13,14,15,16,19,22,17,20,23,25]] 
    
    def preprocess_action(self,action,state=None):
        action = action[..., [0,1,2,3,6,9,4,7,10,12,13,14,15,16,19,22,17,20,23,25]]
        
        if self.delta_action:
            dims = len(self.delta_mask)
            action[..., :dims] -= np.expand_dims(np.where(self.delta_mask, state[..., :dims], 0), axis=-2)
        return action

    def preprocess_prompt(self, prompt) :
        return 'Pack earphone'

    def compute_loss(self, pred, target):
        assert pred.shape == target.shape, "pred/target shapes must match"
        B, T, D = pred.shape
        #print(pred.shape)
        _ensure_indices_valid(D, self.gripper_idx, "gripper_idx")

        # Gripper BCE
        #print(torch.unique( target[:, :, self.gripper_idx]))
        # g_losses = [self.mse(pred[:, :, gi], target[:, :, gi]) for gi in self.gripper_idx]
        # gripper_loss = sum(g_losses) / len(self.gripper_idx) * self.GRIPPER_SCALE

        gripper_loss = (
            self.mse(pred[:, :, self.gripper_idx], target[:, :, self.gripper_idx]) 
        ) * self.GRIPPER_SCALE
        # XYZ position
        pos_loss = (
            self.mse(pred[:, :, self.POS_IDX_1], target[:, :, self.POS_IDX_1]) +
            self.mse(pred[:, :, self.POS_IDX_2], target[:, :, self.POS_IDX_2])
        ) * self.XYZ_SCALE

        # Rotation 6D
        rot_loss = (
            self.mse(pred[:, :, self.ROT_IDX_1], target[:, :, self.ROT_IDX_1]) +
            self.mse(pred[:, :, self.ROT_IDX_2], target[:, :, self.ROT_IDX_2])
        ) * self.ROT_SCALE

        action_loss = (pos_loss + rot_loss + gripper_loss) #/ (self.XYZ_SCALE + self.ROT_SCALE + self.GRIPPER_SCALE)

        return {
            "action_loss": action_loss,
            "position_loss": pos_loss,#/ (self.XYZ_SCALE + self.ROT_SCALE + self.GRIPPER_SCALE),
            "rotate6D_loss": rot_loss,#/ (self.XYZ_SCALE + self.ROT_SCALE + self.GRIPPER_SCALE),
            "gripper_loss": gripper_loss,#/ (self.XYZ_SCALE + self.ROT_SCALE + self.GRIPPER_SCALE),
        }
    
    def compute_latent_action_loss(self, pred, target):
        assert pred.shape == target.shape, "pred/target shapes must match"
        action_loss = self.mse(pred, target)  * self.SCALE
        return {
            "action_loss": action_loss,
        }
    
    def preprocess(self, proprio, action, mode="train"):
        """Zero-out gripper channels in proprio/action."""
        proprio_m = proprio.clone()
        action_m = action.clone()
        # proprio_m[..., self.gripper_idx] = 0.0
        # action_m[..., self.gripper_idx] = 0.0
        # if max_state_dim > 0:
        #     proprio_m[..., max_state_dim:] = 0.0
        # else :
        #     proprio_m[..., self.max_action_dim:] = 0.0
        action_m[..., self.max_action_dim:] = 0.0
        return proprio_m, action_m

    def preprocess_gt(self, action, mode="train"):
        """Zero-out gripper channels in proprio/action."""
        EPSILON = 5e-2

        # g = action[..., self.gripper_idx]

        # action[..., self.gripper_idx] = torch.where(
        #     g > -1.5 - EPSILON,
        #     torch.ones_like(g),
        #     torch.zeros_like(g)
        # )
        #print(action[..., self.gripper_idx].shape,torch.sum(action[..., self.gripper_idx]))
        return action

    def postprocess(self, action: torch.Tensor,state = None) -> torch.Tensor:
        """Apply sigmoid to gripper logits."""
        if self.delta_action :
            dims = len(self.delta_mask)
            action[..., :dims] += np.expand_dims(np.where(self.delta_mask, state[..., :dims], 0), axis=-2)
        # if action.size(-1) > max(self.gripper_idx):
        #     action[..., self.gripper_idx] =  (-3) * (1 - torch.sigmoid(action[..., self.gripper_idx]))
        return action





# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "BaseActionSpace",
    "build_action_space",
    "register_action",
    "EE6DActionSpace",
    "EE6D_RobotwinActionSpace"
    "EEQuat_EarActionSpace"
    "EE6D_EarActionSpace"
    "ACTION_REGISTRY",
]
