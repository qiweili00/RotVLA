# RotVLA evaluation (client / server)

A lightweight client/server evaluation for RotVLA. A single inference
server (`../core/scripts/server.py`) serves both benchmarks; it dispatches on
the `action_space` recorded in the checkpoint config (`ee6d_robotwin` /
`ee6d_libero`). The launcher here starts one server replica per GPU, and each
benchmark has its own client scheduler that fans work out across those replicas.

```
experiment/
  server.py            # launch one inference replica per GPU (shared)
  run_server.sh
  config/
    robotwin.yaml      # server + RoboTwin task runs
    libero.yaml        # server + LIBERO suites
  RoboTwin/
    client.py          # per-task scheduler
    eval_robotwin.py   # single sapien rollout over HTTP /act
    run_client.sh
  LIBERO/
    client.py          # per-suite scheduler
    libero_client.py   # LIBERO rollout over HTTP /act
    run_client.sh
```

## 1. Start the server

Edit `server.ckpt` in the config to point at your checkpoint directory, then:

```bash
# RoboTwin checkpoint
./run_server.sh config/robotwin.yaml
# or LIBERO checkpoint
./run_server.sh config/libero.yaml
```

The launcher rewrites `server.host` in the config to the local IP so the client
can reach it. Replica `i` listens on `server.port + i`.

## 2. Run a client

RoboTwin (needs the simulator env with `sapien`):

```bash
export ROBOTWIN_ROOT=/path/to/RoboTwin      # simulator repo (envs/task_config/description)
cd RoboTwin && ./run_client.sh ../config/robotwin.yaml [num_client_gpus] [tasks_per_gpu]
```

LIBERO (needs the LIBERO env):

```bash
export LIBERO_ROOT=/path/to/LIBERO          # repo containing the `libero` package
cd LIBERO && ./run_client.sh ../config/libero.yaml
```

External simulator/benchmark locations are resolved from the `ROBOTWIN_ROOT` and
`LIBERO_ROOT` environment variables (falling back to hard-coded dev paths).
Set them for your own checkout.
