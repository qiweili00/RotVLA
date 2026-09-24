#!/usr/bin/env python3
"""LIBERO client with Evo-1 environment lifecycle and RotVLA action semantics.

The environment is created once per task and reused across that task's
episodes, matching Evo-1.  The model-side action layout remains this
repository's ``ee6d_libero`` layout: xyz + 6D rotation + one gripper channel.
"""

import argparse
import logging
import os
import pathlib
import sys

import imageio
import json_numpy
import numpy as np
import requests
import robosuite.utils.transform_utils as Trans

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Path to a LIBERO benchmark checkout (the repo containing the `libero` package).
# Set the LIBERO_ROOT environment variable to point at your checkout.
LIBERO_ROOT = os.environ.get("LIBERO_ROOT", "")
if LIBERO_ROOT:
    sys.path.append(LIBERO_ROOT)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


EPS = 1e-6
LIBERO_DUMMY_ACTION = [0.0] * 7
MODEL_ACTION_SPACE = "ee6d_libero"
MODEL_ACTION_DIM = 10


def mat_to_rotate6d(rotation_matrix):
    rotation_matrix = np.asarray(rotation_matrix)
    if rotation_matrix.ndim != 2 or rotation_matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 rotation matrix, got {rotation_matrix.shape}")
    return np.concatenate([rotation_matrix[:3, 0], rotation_matrix[:3, 1]], axis=-1)


def rotate6d_to_axis_angle(rotation6d):
    """Convert a batch of 6D rotations to MuJoCo axis-angle vectors."""
    rotation6d = np.asarray(rotation6d)
    single = rotation6d.ndim == 1
    if single:
        rotation6d = rotation6d[None, :]
    if rotation6d.ndim != 2 or rotation6d.shape[1] != 6:
        raise ValueError(f"Expected rotation shape [N, 6], got {rotation6d.shape}")

    a1 = rotation6d[:, 0:3]
    a2 = rotation6d[:, 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + EPS)
    b2_orthogonal = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2_orthogonal / (np.linalg.norm(b2_orthogonal, axis=-1, keepdims=True) + EPS)
    b3 = np.cross(b1, b2, axis=-1)
    matrices = np.stack([b1, b2, b3], axis=-1)

    axis_angles = []
    for matrix in matrices:
        axis_angles.append(Trans.quat2axisangle(Trans.mat2quat(matrix)))
    result = np.stack(axis_angles, axis=0)
    return result[0] if single else result


def get_libero_env(task, resolution=256, seed=None):
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language


def obs_to_payload(obs, prompt):
    """Build the payload expected by this repository's RotVLA server."""
    image0 = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    image1 = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    state = np.concatenate([obs["robo_pos"], obs["robo_ori"], np.array([0.0])])
    return {
        "language_instruction": prompt,
        "proprio": json_numpy.dumps(state),
        "image0": json_numpy.dumps(image0),
        "image1": json_numpy.dumps(image1),
    }


def save_video(frames, filename, save_dir):
    if not frames:
        return
    os.makedirs(save_dir, exist_ok=True)
    imageio.mimsave(os.path.join(save_dir, filename), frames, fps=30)


def convert_actions(actions):
    """Convert RotVLA ee6d_libero actions to robosuite's 7D action format."""
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] < MODEL_ACTION_DIM:
        raise ValueError(
            f"Expected [T, >=10] actions for {MODEL_ACTION_SPACE}, got {actions.shape}"
        )
    return np.concatenate(
        [
            actions[:, :3],
            rotate6d_to_axis_angle(actions[:, 3:9]),
            actions[:, 9:10],
        ],
        axis=-1,
    )


def run_task_suite(act_url, args, log, task_suite_name, max_steps):
    task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
    total_success = 0
    total_episodes = 0
    total_steps = 0

    log.info("=========================== Start task suite %s ============================", task_suite_name)
    print(f"Numbers of tasks: {task_suite.n_tasks}")

    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        task_episodes = min(args.num_episodes, len(initial_states))
        env, task_description = get_libero_env(task, resolution=256, seed=args.SEED)
        task_success = 0

        try:
            log.info("========= Start task %d: %s =========", task_id + 1, task_description)
            # Evo-1 creates one env per task and performs this reset sequence for
            # every episode before restoring the benchmark initial state.
            for ep in range(task_episodes):
                print(f"\n===== Task {task_id} | Episode {ep + 1} =====")
                env.reset()
                obs = env.set_init_state(initial_states[ep])
                done = False

                for _ in range(10):
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)

                for robot in env.env.robots:
                    robot.controller.use_delta = False

                episode_done = False
                episode_failed = False
                frames = []
                executed_policy_steps = 0

                for policy_step in range(max_steps):
                    executed_policy_steps += 1
                    obs["robo_ori"] = mat_to_rotate6d(
                        env.env.robots[0].controller.ee_ori_mat
                    )
                    obs["robo_pos"] = env.env.robots[0].controller.ee_pos
                    payload = obs_to_payload(obs, str(task_description))

                    response = None
                    try:
                        response = requests.post(act_url, json=payload, timeout=30)
                        response.raise_for_status()
                        actions = convert_actions(response.json()["action"])
                    except Exception as exc:
                        print(
                            f"Request failed: {exc}, "
                            f"content: {getattr(response, 'text', '')}"
                        )
                        episode_failed = True
                        break

                    for action in actions[: args.horizon]:
                        action = action.tolist()
                        action[6] = -1 if action[6] > 0.5 else 1
                        try:
                            obs, reward, done, info = env.step(action[:7])
                        except ValueError as exc:
                            print(f"Invalid action: {exc}")
                            episode_failed = True
                            break

                        frames.append(
                            np.hstack(
                                [
                                    np.rot90(obs["agentview_image"], 2),
                                    np.rot90(obs["robot0_eye_in_hand_image"], 2),
                                ]
                            )
                        )
                        print(
                            f"[Step {policy_step}] reward={reward:.2f}, done={done}"
                        )
                        if done:
                            print("Task completed")
                            episode_done = True
                            break

                    if episode_done or episode_failed:
                        break

                save_video(
                    frames,
                    f"task{task_id + 1}_episode{ep + 1}_{int(done)}.mp4",
                    os.path.join(args.video_root, args.run_name, task_suite_name),
                )
                if episode_done:
                    task_success += 1
                    total_success += 1
                    total_steps += executed_policy_steps
                    log.info("Task %d | Episode %d: Success", task_id, ep + 1)
                else:
                    log.info("Task %d | Episode %d: Fail", task_id, ep + 1)

            total_episodes += task_episodes
            log.info(
                "========= Task %d Summary: %d/%d Successful =========",
                task_id + 1,
                task_success,
                task_episodes,
            )
        finally:
            env.close()

    log.info("========= Overall Task Summary =========")
    log.info("Total Successful Episodes: %d/%d", total_success, total_episodes)
    if total_episodes:
        log.info("Rate: %.2f", total_success * 100.0 / total_episodes)
        log.info("Average Steps: %.2f", total_steps / total_episodes)


def main():
    parser = argparse.ArgumentParser(description="LIBERO client with Evo-1 env lifecycle")
    parser.add_argument("--server_host", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=8000)
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--max_steps", type=int, nargs="+", default=[25, 25, 25, 95])
    parser.add_argument(
        "--task_suites",
        nargs="+",
        default=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
    )
    parser.add_argument("--ckpt_name", default="libero_old")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--SEED", type=int, default=42)
    parser.add_argument("--run_name", default="run0",
                        help="Sub-directory name under <video_root> for this run.")
    parser.add_argument(
        "--video_root",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "log_file"),
    )
    args = parser.parse_args()

    if len(args.max_steps) != len(args.task_suites):
        parser.error("--max_steps must provide one value for each --task_suites entry")
    if args.horizon <= 0 or args.num_episodes <= 0:
        parser.error("--horizon and --num_episodes must be positive")

    args.log_file = os.path.join(
        args.video_root,
        args.run_name,
        f"{args.ckpt_name}_{args.task_suites[0]}.txt",
    )
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(args.log_file, mode="a"), logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)
    act_url = f"http://{args.server_host}:{args.server_port}/act"
    print(f"Using RotVLA server at {act_url}; action_space={MODEL_ACTION_SPACE}")

    for task_suite_name, max_steps in zip(args.task_suites, args.max_steps):
        run_task_suite(act_url, args, log, task_suite_name, max_steps)


if __name__ == "__main__":
    main()
