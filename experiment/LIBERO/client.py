#!/usr/bin/env python3
"""Schedule LIBERO suite evaluation across HTTP replicas.

LIBERO is four task suites (spatial / object / goal / long). Each suite is a
job run by ``libero_client.py`` against one RotVLA ``/act`` replica launched by
``../server.py``. The client itself needs no GPU: by default every suite is
launched in parallel (one process per suite), and slot i talks to server
replica ``i % gpu_num``.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import select
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CONFIG = SCRIPT_DIR.parent / "config" / "libero.yaml"

DEFAULT_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
DEFAULT_MAX_STEPS = [25, 25, 25, 95]


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
    parser.add_argument("--max-parallel", type=int, default=None,
                        help="Cap concurrent suite processes. Default: run all suites in parallel.")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()

    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    server = config["server"]
    libero = dict(config.get("libero", {}))

    gpu_num = int(server.get("gpu_num", 1))
    base_port = int(server["port"])
    server_host = server["host"]
    server_ports = [base_port + index for index in range(gpu_num)]

    suites = list(libero.get("task_suites", DEFAULT_SUITES))
    max_steps = list(libero.get("max_steps", DEFAULT_MAX_STEPS))
    if len(max_steps) != len(suites):
        raise ValueError(f"libero.max_steps has {len(max_steps)} entries, expected {len(suites)}")
    num_episodes = int(libero.get("num_episodes", 10))
    horizon = int(libero.get("horizon", 14))
    seed = int(libero.get("seed", 42))
    run_name = str(libero.get("run_name", "run0"))

    output_root = Path(config.get("output_root", "eval_result/libero"))
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    tag = ckpt_tag(server)
    video_root = output_root / tag
    video_root.mkdir(parents=True, exist_ok=True)

    job_queue: deque[dict] = deque(
        {"suite": suite, "max_steps": steps}
        for suite, steps in zip(suites, max_steps)
    )
    total_jobs = len(job_queue)
    num_slots = total_jobs if args.max_parallel is None else min(args.max_parallel, total_jobs)
    if num_slots < 1:
        raise ValueError("--max-parallel must be >= 1")
    print(f"[client] Config: {config_path}")
    print(f"[client] Suites: {suites}")
    print(f"[client] Concurrent workers: {num_slots}, Server replicas: {gpu_num}")
    print(f"[client] Output root: {video_root}/")

    eval_script = (SCRIPT_DIR / "libero_client.py").resolve()
    workers: list[tuple | None] = [None] * num_slots
    completed = failed = 0
    pending = job_queue

    def launch_next(slot: int) -> bool:
        if not pending:
            return False
        job = pending.popleft()
        replica = slot % len(server_ports)
        port = server_ports[replica]
        environment = os.environ.copy()
        environment.update(PYTHONUNBUFFERED="1")
        command = [
            sys.executable, "-u", str(eval_script),
            "--server_host", str(server_host), "--server_port", str(port),
            "--task_suites", job["suite"],
            "--max_steps", str(job["max_steps"]),
            "--num_episodes", str(num_episodes),
            "--horizon", str(horizon),
            "--SEED", str(seed),
            "--ckpt_name", tag,
            "--run_name", run_name,
            "--video_root", str(video_root),
        ]
        process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert process.stdout is not None
        set_nonblocking(process.stdout.fileno())
        workers[slot] = (process, job, replica)
        print(f"    [replica {replica}] -> {job['suite']} (max_steps={job['max_steps']})")
        return True

    for slot in range(num_slots):
        launch_next(slot)
    while any(worker is not None for worker in workers) or pending:
        read_pipes = [worker[0].stdout for worker in workers if worker is not None and worker[0].stdout]
        if read_pipes:
            readable, _, _ = select.select(read_pipes, [], [], 0.5)
            for pipe in readable:
                slot = next(index for index, worker in enumerate(workers) if worker and worker[0].stdout is pipe)
                process, job, replica = workers[slot]
                data = pipe.read(4096)
                if data:
                    text = data.decode("utf-8", errors="replace")
                    for line in text.splitlines(keepends=True):
                        sys.stdout.write(f"[replica {replica}|{job['suite']}] {line}")
                    sys.stdout.flush()
        else:
            time.sleep(0.5)
        for slot, worker in enumerate(workers):
            if worker is None:
                continue
            process, job, replica = worker
            return_code = process.poll()
            if return_code is None:
                continue
            if process.stdout:
                remaining = process.stdout.read()
                if remaining:
                    sys.stdout.write(remaining.decode("utf-8", errors="replace"))
            if return_code == 0:
                completed += 1
                print(f"    [replica {replica}] done: {job['suite']} ({completed}/{total_jobs})")
            else:
                failed += 1
                print(f"    [replica {replica}] FAILED: {job['suite']} (code {return_code})")
            workers[slot] = None
            launch_next(slot)
    print(f"\n{'=' * 60}\nAll done. Completed: {completed}, Failed: {failed}, Total: {total_jobs}\n{'=' * 60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
