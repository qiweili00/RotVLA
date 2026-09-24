# LIBERO evaluation

`client.py` schedules the four LIBERO suites (`libero_spatial`, `libero_object`,
`libero_goal`, `libero_10`) — one suite per GPU slot. Each slot runs
`libero_client.py` for that suite against a RotVLA `/act` replica, iterating its
tasks (10 each) serially.

Run from the LIBERO environment, after starting the server (see `../README.md`):

```bash
export LIBERO_ROOT=/path/to/LIBERO      # repo containing the `libero` package
./run_client.sh ../config/libero.yaml
```

By default every suite runs in parallel (one process each). Pass `--max-parallel N`
to `client.py` to cap how many run at once.

Suites, per-suite `max_steps`, `num_episodes`, `horizon` and `seed` come from the
`libero` block of the config. Videos and logs are written under
`output_root/<tag>/`.

`libero_client.py` can also be run standalone against a single server:

```bash
python libero_client.py --server_host 127.0.0.1 --server_port 8021 \
    --task_suites libero_spatial --max_steps 25 --num_episodes 10
```
