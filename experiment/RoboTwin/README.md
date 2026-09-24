# RoboTwin evaluation

`client.py` schedules RoboTwin tasks across GPU slots; each task runs one
`eval_robotwin.py` sapien rollout that talks to a RotVLA `/act` replica.

Run from the simulator environment (the one with `sapien`), after starting the
server (see `../README.md`):

```bash
export ROBOTWIN_ROOT=/path/to/RoboTwin
./run_client.sh ../config/robotwin.yaml --num-client-gpus 8 --tasks-per-gpu 3
```

`ROBOTWIN_ROOT` points at the RoboTwin simulator repo (`envs`, `task_config`,
`description`). The client reads `defaults` / `eval_runs` from the config:
each `eval_runs` entry runs the task list (default: all 50 tasks) with the given
`params` (e.g. `task_config`, `replan_steps`). Results are written under
`output_root/<tag>/<run_name>/`.
