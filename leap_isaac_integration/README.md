# LEAP + Isaac Lab Integration

This folder wires your existing Isaac Lab setup with `LEAP_Hand_Isaac_Lab` and launches the official LEAP training task from the upstream README.

## Scripts

- `setup_and_train_leap.sh`
  - Activates `isaac_fresh`
  - Installs `source/LEAP_Isaaclab` in editable mode
  - Runs:
    - `python scripts/rl_games/train.py --task Isaac-Reorient-Cube-Leap --headless --video`
  - Video defaults:
    - `VIDEO_INTERVAL=2000` steps
    - `VIDEO_LENGTH=200` steps
  - Video location:
    - `LEAP_Hand_Isaac_Lab/logs/rl_games/<run_name>/<timestamp>/videos/train`

- `master_leap_train.sh`
  - SLURM wrapper that calls `setup_and_train_leap.sh`

## Local run

```bash
cd ~/yash/leap_isaac_integration
bash setup_and_train_leap.sh
```

Override video cadence:

```bash
VIDEO_INTERVAL=4000 VIDEO_LENGTH=300 bash setup_and_train_leap.sh
```

## SLURM run

**Do not run `setup_and_train_leap.sh` or `resume_train_leap.sh` on the login node.** Isaac Sim needs a GPU; the login node has no `libcuda` / `/dev/nvidia*`, which produces errors like `libcuda.so.1: cannot open shared object file`.

Fresh training on a compute node:

```bash
cd ~/yash/leap_isaac_integration
sbatch master_leap_train.sh
```

Resume (same defaults as `resume_train_leap.sh`; override `RUN_NAME` / `CKPT` if needed):

```bash
cd ~/yash/leap_isaac_integration
sbatch master_leap_resume.sh
# optional:
# RUN_NAME=2026-05-09_20-00-57 CKPT=/path/to/nn/leap_hand_reorient.pth sbatch master_leap_resume.sh
```

## Resume after a time limit (or any stop)

Checkpoints live under `LEAP_Hand_Isaac_Lab/logs/rl_games/leap_hand_reorient/<run_timestamp>/nn/`.

**Two file types (RL-Games):** `leap_hand_reorient.pth` is updated only when **mean train reward beats the previous best**—it is **not** “latest epoch.” Files like `last_leap_hand_reorient_ep_4000_rew_*.pth` are **periodic snapshots** (every `save_frequency` epochs, e.g. 200). For resume you usually want the **latest `last_*_ep_*`** so the epoch continues from where you trained (see `CHECKPOINT_SELECT` in `resume_train_leap.sh`).

Resume into the **same** run folder with:

```bash
cd ~/yash/leap_isaac_integration
RUN_NAME=2026-05-09_20-00-57 bash resume_train_leap.sh
```

Override checkpoint explicitly:  
`CKPT=/path/to/nn/last_leap_hand_reorient_ep_4100_rew_....pth RUN_NAME=<dir_name> bash resume_train_leap.sh`  
Or use `CHECKPOINT_SELECT=best` to force `leap_hand_reorient.pth` (reward-best, can be far behind in epoch).

Default `resume_train_leap.sh` chooses **automatically**: newest `last_*_ep_*` milestone, else fallback to `leap_hand_reorient.pth`.

Hydra treats `full_experiment_name` as an extra field, so resume uses **`+agent.params.config.full_experiment_name=…`** internally (fixes `Could not override ... Key full_experiment_name is not in struct`).

Increase wall time in `master_leap_train.sh` (`#SBATCH --time=...`) so the job can reach `max_epochs` in `rl_games_ppo_cfg.yaml`.

## `./isaaclab.sh --new` vs training

- **`./isaaclab.sh -p script.py`** runs Python with Isaac Sim / Lab (what `setup_and_train_leap.sh` uses). This is how you train.
- **`./isaaclab.sh --new`** only runs the **template generator** for a new project or task; it is not required before training.
- **One-time setup** for Isaac Lab is normally: install Isaac Sim in the conda env, then `./isaaclab.sh -i` (install Lab extensions + RL extras), activate that env, then run training scripts. Your `setup_and_train_leap.sh` already activates `isaac_fresh`, `pip install -e` the LEAP extension, and calls `isaaclab.sh -p .../train.py`.
