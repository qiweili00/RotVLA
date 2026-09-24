#!/usr/bin/env python3
"""Schedule RoboTwin evaluation tasks across local GPUs and HTTP replicas.

Each task is run by ``eval_robotwin.py`` (a single sapien rollout) talking to a
RotVLA ``/act`` replica launched by ``../server.py``. Tasks are pulled from a
queue and dispatched to ``num_client_gpus * tasks_per_gpu`` concurrent slots;
slot i uses server replica ``i % gpu_num``.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import select
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CONFIG = SCRIPT_DIR.parent / "config" / "robotwin.yaml"

DEFAULT_TASKS = [
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size",
    "click_alarmclock", "click_bell", "grab_roller", "handover_block", "handover_mic",
    "hanging_mug", "lift_pot", "move_can_pot", "move_pillbottle_pad",
    "move_playingcard_away", "move_stapler_pad", "open_laptop", "open_microwave",
    "pick_diverse_bottles", "pick_dual_bottles", "place_a2b_left", "place_a2b_right",
    "place_bread_basket", "place_bread_skillet", "place_burger_fries", "place_can_basket",
    "place_cans_plasticbox", "place_container_plate", "place_dual_shoes", "place_empty_cup",
    "place_fan", "place_mouse_pad", "place_object_basket", "place_object_scale",
    "place_object_stand", "place_phone_stand", "place_shoe", "dump_bin_bigbin",
    "press_stapler", "put_object_cabinet", "rotate_qrcode", "scan_object",
    "shake_bottle_horizontally", "shake_bottle", "stack_blocks_three", "stack_blocks_two",
    "stack_bowls_three", "stack_bowls_two", "stamp_seal", "turn_switch", "put_bottles_dustbin",
]


def detect_gpu_count() -> int:
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
        count = len([line for line in result.stdout.splitlines() if line.strip()])
    except OSError:
        count = 0
    return max(count, 1)


def set_nonblocking(fd: int) -> None:
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)


def ckpt_tag(server: dict) -> str:
    checkpoint = str(server.get("ckpt", "rotvla")).rstrip("/")
    parts = checkpoint.split("/")
    config_name = server.get("config", "rotvla")
    return f"{config_name}_{parts[-2]}_{parts[-1]}" if len(parts) >= 2 else str(config_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--num-client-gpus", type=int, default=None)
    parser.add_argument("--tasks-per-gpu", type=int, default=None)
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()

    try:
        import sapien  # noqa: F401
    except ImportError:
        raise SystemExit(
            f"\n[client] ERROR: `sapien` not found in this Python environment ({sys.executable}).\n"
            "The eval client runs the RoboTwin simulator and needs sapien.\n"
            "Activate the simulator environment first, for example: conda activate RoboTwin\n"
        )

    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    server = config["server"]
    num_client_gpus = args.num_client_gpus or detect_gpu_count()
    if num_client_gpus < 1:
        raise ValueError("--num-client-gpus must be >= 1")
    gpu_num = int(server.get("gpu_num", 1))
    base_port = int(server["port"])
    server_host = server["host"]
    server_ports = [base_port + index for index in range(gpu_num)]

    defaults = dict(config.get("defaults", {}))
    configured_tpg = defaults.pop("tasks_per_gpu", None)
    tasks_per_gpu = args.tasks_per_gpu if args.tasks_per_gpu is not None else int(configured_tpg or 1)
    if tasks_per_gpu < 1:
        raise ValueError(f"tasks_per_gpu must be >= 1, got {tasks_per_gpu}")

    output_root = Path(config.get("output_root", "eval_result/robotwin"))
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    tag = ckpt_tag(server)
    task_queue: deque[dict] = deque()
    for run in config.get("eval_runs", []):
        run_name = run["name"]
        run_tasks = run.get("tasks", defaults.get("tasks", DEFAULT_TASKS))
        merged_params = {key: value for key, value in defaults.items() if key != "tasks"}
        merged_params.update(run.get("params", {}))

        base_dir = output_root / tag / run_name
        base_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, base_dir / "eval_config.yaml")
        for task in run_tasks:
            task_queue.append({"task": task, "output_dir": base_dir / task,
                               "params": merged_params, "run_name": run_name})

    total_jobs = len(task_queue)
    num_slots = num_client_gpus * tasks_per_gpu
    slot_gpu = [slot % num_client_gpus for slot in range(num_slots)]
    print(f"[client] Config: {config_path}")
    print(f"[client] Total jobs: {total_jobs} (from {len(config.get('eval_runs', []))} eval_runs)")
    print(f"[client] GPUs: {num_client_gpus}, tasks_per_gpu: {tasks_per_gpu}, "
          f"concurrent workers: {num_slots}, Server replicas: {gpu_num}")
    print(f"[client] Output root: {output_root / tag}/")

    eval_script = (SCRIPT_DIR / "eval_robotwin.py").resolve()
    workers: list[tuple | None] = [None] * num_slots
    completed = failed = 0
    pending = task_queue

    def launch_next(slot: int) -> bool:
        if not pending:
            return False
        job = pending.popleft()
        gpu_id = slot_gpu[slot]
        port = server_ports[slot % len(server_ports)]
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=str(gpu_id), PYTHONUNBUFFERED="1")
        command = [sys.executable, "-u", str(eval_script), "--host", str(server_host),
                   "--port", str(port), "--task_name", job["task"],
                   "--eval_output_dir", str(job["output_dir"])]
        for key, value in job["params"].items():
            if key == "tasks":
                continue
            command.extend([f"--{key}", str(value).lower() if isinstance(value, bool) else str(value)])
        output_dir = Path(job["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = (output_dir / "eval.log").open("w", encoding="utf-8")
        process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert process.stdout is not None
        set_nonblocking(process.stdout.fileno())
        workers[slot] = (process, job, log_file, gpu_id)
        print(f"    [GPU {gpu_id}] -> {job['run_name']}/{job['task']}")
        return True

    for slot in range(num_slots):
        launch_next(slot)
    while any(worker is not None for worker in workers) or pending:
        read_pipes = [worker[0].stdout for worker in workers if worker is not None and worker[0].stdout]
        if read_pipes:
            readable, _, _ = select.select(read_pipes, [], [], 0.5)
            for pipe in readable:
                slot = next(index for index, worker in enumerate(workers) if worker and worker[0].stdout is pipe)
                process, job, log_file, gpu_id = workers[slot]
                data = pipe.read(4096)
                if data:
                    text = data.decode("utf-8", errors="replace")
                    for line in text.splitlines(keepends=True):
                        sys.stdout.write(f"[GPU {gpu_id}|{job['run_name']}/{job['task']}] {line}")
                        log_file.write(line)
                    sys.stdout.flush(); log_file.flush()
        else:
            time.sleep(0.5)
        for slot, worker in enumerate(workers):
            if worker is None:
                continue
            process, job, log_file, gpu_id = worker
            return_code = process.poll()
            if return_code is None:
                continue
            remaining = process.stdout.read() if process.stdout else b""
            if remaining:
                log_file.write(remaining.decode("utf-8", errors="replace"))
            log_file.close()
            if return_code == 0:
                completed += 1
                print(f"    [GPU {gpu_id}] done: {job['run_name']}/{job['task']} ({completed}/{total_jobs})")
            else:
                failed += 1
                print(f"    [GPU {gpu_id}] FAILED: {job['run_name']}/{job['task']} (code {return_code})")
            workers[slot] = None
            launch_next(slot)
    print(f"\n{'=' * 60}\nAll done. Completed: {completed}, Failed: {failed}, Total: {total_jobs}\n{'=' * 60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
