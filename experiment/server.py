#!/usr/bin/env python3
"""Launch one RotVLA HTTP inference server replica per configured GPU.

The same server (``core/scripts/server.py``) drives both the RoboTwin and the
LIBERO evaluation clients — it dispatches on the ``action_space`` recorded in
the checkpoint config (``ee6d_robotwin`` / ``ee6d_libero``). Point this launcher
at either ``config/robotwin.yaml`` or ``config/libero.yaml``; only the ``server``
mapping is read here.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / "config" / "robotwin.yaml"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config.get("server"), dict):
        raise ValueError(f"Config must contain a 'server' mapping: {path}")
    return config


def detect_local_ip() -> str:
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG,
                        help=f"YAML config (default: {DEFAULT_CONFIG})")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    print(f"[server] Config: {config_path}")

    config = load_config(config_path)
    server = config["server"]
    gpu_num = int(server.get("gpu_num", 1))
    if gpu_num < 1:
        raise ValueError(f"server.gpu_num must be >= 1, got {gpu_num}")
    gpu_ids = [int(gpu) for gpu in server.get("gpu_ids", range(gpu_num))]
    if len(gpu_ids) < gpu_num:
        raise ValueError(f"server.gpu_ids has {len(gpu_ids)} entries, expected {gpu_num}")
    base_port = int(server["port"])
    checkpoint = server["ckpt"]
    time_step = int(server.get("time_step", 20))

    local_ip = detect_local_ip()
    server["host"] = local_ip
    with config_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file, default_flow_style=False, sort_keys=False)
    print(f"[server] Updated config host to: {local_ip}")
    print(f"[server] Checkpoint (ckpt_dir): {checkpoint}")
    print(f"[server] GPU IDs: {gpu_ids[:gpu_num]}, base port: {base_port}, time_step: {time_step}\n")

    server_script = PROJECT_ROOT / "core" / "scripts" / "server.py"
    if not server_script.is_file():
        raise SystemExit(f"server.py not found at {server_script}")

    processes: list[tuple[str, subprocess.Popen]] = []
    for index, gpu_id in enumerate(gpu_ids[:gpu_num]):
        port = base_port + index
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=str(gpu_id), PYTHONUNBUFFERED="1")
        command = [
            sys.executable, "-u", str(server_script),
            "--host", "0.0.0.0",
            "--port", str(port),
            "--ckpt_dir", str(checkpoint),
            "--time_step", str(time_step),
        ]
        print(f"  [replica {index}] GPU {gpu_id}, port {port}")
        processes.append((f"replica_{index}", subprocess.Popen(command, env=environment)))

    def stop_all(*_signals: object) -> None:
        print("\n[server] Stopping all...")
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            process.wait()
        print("[server] All stopped.")

    signal.signal(signal.SIGTERM, stop_all)
    print(f"\n[server] {len(processes)} process(es) started. Press Ctrl+C to stop all.\n")
    try:
        for _, process in processes:
            process.wait()
    except KeyboardInterrupt:
        stop_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
