#!/usr/bin/env python3
"""
RoboTwin evaluation client for the RotVLA HTTP policy server.

An openpi-style eval loop (take_action_cnt / step_lim / eval_success) driving
the RotVLA ``/act`` HTTP protocol (json_numpy) with ee6d / rotate6d action
decoding. One sapien rollout per invocation; scheduled by client.py.

The RoboTwin simulator repo (envs / task_config / description) is located via
the ROBOTWIN_ROOT environment variable, falling back to the path below.

Usage:
    ROBOTWIN_ROOT=/path/to/RoboTwin python eval_robotwin.py \
        --host localhost --port 8011 \
        --task_name beat_block_hammer --task_config demo_clean
"""

import sys
import os
from pathlib import Path

# RoboTwin simulator root: env / task_config / description are loaded from here.
# Set the ROBOTWIN_ROOT environment variable to point at your checkout.
_robotwin_root = os.environ.get("ROBOTWIN_ROOT")
if not _robotwin_root:
    raise SystemExit(
        "ROBOTWIN_ROOT is not set. Point it at your RoboTwin simulator checkout, "
        "e.g. export ROBOTWIN_ROOT=/path/to/RoboTwin"
    )
ROBOTWIN_ROOT = Path(_robotwin_root)
if str(ROBOTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_ROOT))
os.chdir(ROBOTWIN_ROOT)
sys.path.append(str(ROBOTWIN_ROOT / "description" / "utils"))

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import argparse
import json
import random
import traceback
from datetime import datetime
from typing import Optional

import cv2  # noqa: F401  (kept for parity / optional image ops)
import imageio
import numpy as np
import json_numpy
import requests
import torch
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from generate_episode_instructions import generate_episode_descriptions


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_decorator(task_name):
    import importlib
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception:
        raise SystemExit("No Task")
    return env_instance


# ---------------------------------------------------------------------------
# Quaternion / rotation helpers (RotVLA uses scipy xyzw ordering)
# ---------------------------------------------------------------------------

def quat_to_rotate6D(q: np.ndarray) -> np.ndarray:
    """xyzw quaternion -> 6D rotation (first two matrix columns)."""
    return R.from_quat(q).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))


def rotate6D_to_quat(v6: np.ndarray) -> np.ndarray:
    """6D rotation -> xyzw quaternion (Gram-Schmidt orthonormalization)."""
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError("Last dimension must be 6 (got %s)" % (v6.shape[-1],))
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)
    return R.from_matrix(rot_mats).as_quat()


def _resize_rgb(image: np.ndarray, size_wh: tuple) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _center_crop_rgb(image: np.ndarray, crop_ratio: float = 1.0) -> np.ndarray:
    image = image.astype(np.uint8)
    if not 0 < crop_ratio <= 1:
        raise ValueError("crop_ratio must be in (0, 1].")
    if crop_ratio == 1:
        return image

    height, width = image.shape[:2]
    crop_h = int(height * crop_ratio)
    crop_w = int(width * crop_ratio)
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    cropped = image[top:top + crop_h, left:left + crop_w]
    resized = Image.fromarray(cropped, mode="RGB").resize((width, height), resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)

def _load_task_args(task_name, task_config):
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(et):
        robot_file = _embodiment_types[et]["file_path"]
        if robot_file is None:
            raise ValueError("No embodiment files")
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    # Single (dual-arm) embodiment only. The heterogeneous 3-item form used for
    # ARX/Franka mixes is intentionally not supported in this public client.
    if len(embodiment_type) != 1:
        raise ValueError(
            f"Only single-embodiment task configs are supported, got "
            f"embodiment={embodiment_type!r} in {task_config}."
        )
    args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
    args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
    args["dual_arm_embodied"] = True

    def get_embodiment_config(robot_file):
        robot_config_file = os.path.join(robot_file, "config.yml")
        with open(robot_config_file, "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


# ---------------------------------------------------------------------------
# RotVLA HTTP policy (ee6d, remote HTTP /act)
# ---------------------------------------------------------------------------

class RotVLAHttpPolicy:
    """Policy that connects to the RotVLA HTTP `/act` server (json_numpy).

    The server is fed a 20-dim proprio vector and head/left/right RGB images,
    and returns actions of shape [T, 24] laid out as:
        left:  xyz(3) rotate6d(6) gripper(1)          -> dims 0..9
        right: xyz(3) rotate6d(6) gripper(1)          -> dims 10..19
    (dims 20..23 are unused padding kept for weight compatibility).

    Decoded into RoboTwin ee actions (16-dim):
        [left_xyz(3), left_quat(4), left_grip(1),
         right_xyz(3), right_quat(4), right_grip(1)]
    and executed via take_action(action_type='ee'). The proprio 6D and the
    action quaternion use scipy ordering with NO reordering.
    """

    def __init__(self, host: str, port: int,
                 replan_steps: int = 24,
                 crop_ratio: float = 1.0,
                 gripper_thresh: float = 0.7,
                 send_resize: Optional[int] = None):
        self.url = f"http://{host}:{port}/act"
        self.replan_steps = int(replan_steps)
        self.crop_ratio = float(crop_ratio)
        if not 0 < self.crop_ratio <= 1:
            raise ValueError("crop_ratio must be in (0, 1].")
        self.gripper_thresh = float(gripper_thresh)
        # None -> send images at native resolution (default, lowest risk).
        # An int -> resize head/left/right to (send_resize, send_resize).
        self.send_resize = send_resize
        self.instruction = ""
        self.pending_actions = []
        self.step_count = 0
        print(f"[image preprocess] head_camera_crop_ratio={self.crop_ratio}, "
              f"send_resize={self.send_resize}")
        print(f"[action] replan_steps={self.replan_steps}, "
              f"gripper_thresh={self.gripper_thresh}")

    def set_instruction(self, instruction):
        self.instruction = instruction

    # --- observation -> server payload --------------------------------------

    def _prep_image(self, rgb: np.ndarray, is_head: bool = False) -> np.ndarray:
        if is_head and self.crop_ratio < 1:
            rgb = _center_crop_rgb(rgb, crop_ratio=self.crop_ratio)
        if self.send_resize is not None:
            rgb = _resize_rgb(rgb, (self.send_resize, self.send_resize))
        return np.ascontiguousarray(rgb.astype(np.uint8))

    def _build_proprio(self, observation: dict) -> np.ndarray:
        """20-dim: left[xyz(3) rot6d(6) grip(1)] + right[xyz(3) rot6d(6) grip(1)].

        The endpose quaternion is fed straight into quat_to_rotate6D (scipy)
        with NO reordering, and the gripper uses the RotVLA training convention
        grip = 1 - 2*grip.
        """
        endpose = observation["endpose"]
        left_ee = np.asarray(endpose["left_endpose"], dtype=np.float64)
        right_ee = np.asarray(endpose["right_endpose"], dtype=np.float64)
        left_grip = float(endpose["left_gripper"])
        right_grip = float(endpose["right_gripper"])
        left_grip = 1 - left_grip * 2
        right_grip = 1 - right_grip * 2

        proprio = np.concatenate([
            left_ee[:3],
            quat_to_rotate6D(left_ee[3:7]),
            [left_grip],
            right_ee[:3],
            quat_to_rotate6D(right_ee[3:7]),
            [right_grip],
        ]).astype(np.float32)
        return proprio

    def _infer(self, observation: dict) -> np.ndarray:
        obs_data = observation["observation"]
        head = self._prep_image(obs_data["head_camera"]["rgb"], is_head=True)
        left = self._prep_image(obs_data["left_camera"]["rgb"])
        right = self._prep_image(obs_data["right_camera"]["rgb"])
        proprio = self._build_proprio(observation)

        payload = {
            "language_instruction": self.instruction,
            "proprio": json_numpy.dumps(proprio),
            "image0": json_numpy.dumps(head),
            "image1": json_numpy.dumps(left),
            "image2": json_numpy.dumps(right),
        }
        resp = requests.post(self.url, json=payload, timeout=30)
        resp.raise_for_status()
        return np.asarray(resp.json()["action"], dtype=np.float32)

    def _decode_actions(self, actions: np.ndarray) -> np.ndarray:
        """[T,24] ee6d -> [T,16] RoboTwin ee action.

        The scipy quaternion from rotate6D_to_quat is placed straight into the
        ee action with NO reordering; take_action(ee) consumes this layout.
        """
        left_xyz = actions[:, 0:3]
        left_quat = rotate6D_to_quat(actions[:, 3:9])
        left_grip = (actions[:, 9:10] > self.gripper_thresh).astype(np.float32)

        right_xyz = actions[:, 10:13]
        right_quat = rotate6D_to_quat(actions[:, 13:19])
        right_grip = (actions[:, 19:20] > self.gripper_thresh).astype(np.float32)

        return np.concatenate([
            left_xyz, left_quat, left_grip,
            right_xyz, right_quat, right_grip,
        ], axis=1)

    # --- openpi-style stepping ----------------------------------------------

    def step(self, task_env, observation: Optional[dict]) -> None:
        if not self.pending_actions:
            if observation is None:
                raise ValueError("Observation required when action queue empty.")
            raw = self._infer(observation)
            decoded = self._decode_actions(raw)
            n_exec = min(self.replan_steps, decoded.shape[0])
            self.pending_actions = list(decoded[:n_exec])

        action = self.pending_actions.pop(0)
        task_env.take_action(action, action_type="ee")
        self.step_count += 1

    def reset(self):
        self.pending_actions = []
        self.step_count = 0


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def eval_policy(task_name, TASK_ENV, args, model, st_seed, test_num=100,
                video_save_dir=None):
    print(f"\033[34mTask Name: {task_name}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    now_seed = st_seed
    clear_cache_freq = args["clear_cache_freq"]
    args["eval_mode"] = True

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                seed_everything(now_seed)
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError:
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        try:
            seed_everything(now_seed)
            TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        except (UnStableError, Exception):
            succ_seed -= 1
            TASK_ENV.close_env()
            now_seed += 1
            continue

        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(task_name, episode_info_list, test_num)
        instruction = np.random.choice(results[0]['seen'])
        TASK_ENV.set_instruction(instruction=instruction)
        instruction = TASK_ENV.get_instruction()
        print(f"[instruction] {instruction}")
        model.set_instruction(instruction)

        episode_frames = []
        succ = False
        model.reset()

        pbar = tqdm(total=TASK_ENV.step_lim, desc=f"ep{now_id}", unit="step", dynamic_ncols=True)
        prev_cnt = 0

        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            pbar.update(TASK_ENV.take_action_cnt - prev_cnt)
            prev_cnt = TASK_ENV.take_action_cnt

            observation = None
            if not model.pending_actions:
                observation = TASK_ENV.get_obs()

            if video_save_dir is not None and observation is not None:
                obs_dict = observation["observation"]
                head_rgb = obs_dict["head_camera"]["rgb"]
                episode_frames.append(head_rgb.copy())

            model.step(TASK_ENV, observation)

            if TASK_ENV.eval_success:
                succ = True
                pbar.update(TASK_ENV.take_action_cnt - prev_cnt)
                break

        pbar.update(TASK_ENV.take_action_cnt - prev_cnt)
        pbar.set_postfix(success=succ)
        pbar.close()

        if video_save_dir is not None and episode_frames:
            episode_idx = TASK_ENV.test_num
            video_path = (
                Path(video_save_dir)
                / f"episode{episode_idx}_seed-{now_seed}_success-{str(succ).lower()}.mp4"
            )
            imageio.mimsave(str(video_path), episode_frames, fps=30)

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()
        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | "
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => "
            f"\033[95m{round(TASK_ENV.suc / TASK_ENV.test_num * 100, 1)}%\033[0m, "
            f"current seed: \033[90m{now_seed}\033[0m\n"
        )
        now_seed += 1

    return now_seed, TASK_ENV.suc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RoboTwin eval with RotVLA HTTP server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--task_config", type=str, default="demo_clean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_num_episodes", type=int, default=100)
    parser.add_argument("--replan_steps", type=int, default=24)
    parser.add_argument("--gripper_thresh", type=float, default=0.7)
    parser.add_argument("--send_resize", type=int, default=None,
                        help="If set, resize head/left/right images to this square "
                             "size before sending. Default: send native resolution.")
    parser.add_argument("--save_video", type=str, default="false")
    parser.add_argument("--eval_output_dir", type=str, default=None)
    parser.add_argument("--crop_ratio", type=float, default=1.0)
    cli_args = parser.parse_args()

    eval_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = cli_args.task_name
    task_config = cli_args.task_config

    args = _load_task_args(task_name, task_config)
    args["eval_video_log"] = False
    args["eval_video_save_dir"] = None

    # ee mode reads endpose from the observation; make sure the env emits it
    # even if the task_config didn't request it.
    if not isinstance(args.get("data_type"), dict):
        args["data_type"] = {}
    args["data_type"]["endpose"] = True

    if cli_args.eval_output_dir:
        save_dir = Path(cli_args.eval_output_dir)
        if not save_dir.is_absolute():
            save_dir = Path.cwd() / save_dir
    else:
        save_dir = ROBOTWIN_ROOT / f"eval_result/{task_name}/{task_config}/{eval_ts}"
    save_dir.mkdir(parents=True, exist_ok=True)

    seed = cli_args.seed
    seed_everything(seed)
    TASK_ENV = class_decorator(task_name)

    model = RotVLAHttpPolicy(
        host=cli_args.host,
        port=cli_args.port,
        replan_steps=cli_args.replan_steps,
        crop_ratio=cli_args.crop_ratio,
        gripper_thresh=cli_args.gripper_thresh,
        send_resize=cli_args.send_resize,
    )

    st_seed = 100000 * (1 + seed)
    save_video = cli_args.save_video.lower() in ("true", "1", "yes")
    video_dir = str(save_dir) if save_video else None
    st_seed, suc_num = eval_policy(
        task_name, TASK_ENV, args, model, st_seed,
        test_num=cli_args.eval_num_episodes,
        video_save_dir=video_dir,
    )

    # Save results
    result_file = save_dir / "_result.txt"
    with open(result_file, "w") as f:
        f.write(f"Timestamp: {eval_ts}\n\n")
        f.write(str(suc_num / cli_args.eval_num_episodes))

    # Write perc.json (JSONL) in parent directory (run_name level)
    perc_file = save_dir.parent / "perc.json"
    perc_file.parent.mkdir(exist_ok=True, parents=True)
    with open(perc_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({task_name: round(float(suc_num / cli_args.eval_num_episodes * 100), 4)},
                           ensure_ascii=False) + "\n")

    print(f"\nResults saved to {result_file} and {perc_file}")


if __name__ == "__main__":
    main()
