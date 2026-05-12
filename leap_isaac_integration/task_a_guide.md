## Task A on the GPU cluster (grasp + lift): complete boilerplate

This guide is written to match your existing cluster setup in `leap_isaac_integration/`:

- Conda env: `isaac_fresh`
- Isaac Lab launcher: `~/yash/IsaacLab/isaaclab.sh`
- LEAP repo: `~/yash/LEAP_Hand_Isaac_Lab`

### What you already have

- Task code lives in: `LEAP_Hand_Isaac_Lab/grasp_lift_task/`
- Gym task id: `Isaac-Leap-Grasp-Lift-v0`
- Trainer: RL-Games PPO (config: `LEAP_Hand_Isaac_Lab/grasp_lift_task/agents/rl_games_ppo_cfg.yaml`)
- Cluster launcher script (added): `leap_isaac_integration/task_a_grasp_lift_train.sh`
- SLURM wrapper (added): `leap_isaac_integration/master_task_a_grasp_lift_train.sh`

### Requirement checklist (mapped to files)

1. **Isolated workspace**
   - Folder: `LEAP_Hand_Isaac_Lab/grasp_lift_task/`
   - No existing root files were edited.
2. **New task name**
   - Registered in `LEAP_Hand_Isaac_Lab/grasp_lift_task/__init__.py` as:
     - `Isaac-Leap-Grasp-Lift-v0`
3. **Robot + scene**
   - `LEAP_Hand_Isaac_Lab/grasp_lift_task/leap_hand_lift_env_cfg.py`
   - Robot: LEAP hand (16 dof), floating base at `(0,0,0.5)`
   - Table: static cuboid at `(0,0,0)`
   - Object: 5cm cuboid cube
4. **MDP config**
   - Obs: `q, qd, object pose, palm->object`
   - Action: 16-dim continuous -> PD position targets
   - Implemented in:
     - env cfg (`leap_hand_lift_env_cfg.py`) and env (`leap_hand_lift_env.py`)
5. **Dense reward**
   - Implemented in `LEAP_Hand_Isaac_Lab/grasp_lift_task/leap_hand_lift_env.py`:
     - Reach, Contact, Lift, Torque penalty, Fall penalty
6. **Training setup**
   - PPO YAML: `LEAP_Hand_Isaac_Lab/grasp_lift_task/agents/rl_games_ppo_cfg.yaml`
   - A100-friendly values, AMP enabled
   - `num_envs=1024`, `--headless` used by the launcher script
7. **Headless logging**
   - WandB integrated in `LEAP_Hand_Isaac_Lab/grasp_lift_task/train_rl_games.py`
   - Video highlights via Gym `RecordVideo` wrapper

### Cluster training (SLURM)

Submit:

```bash
cd ~/yash/leap_isaac_integration
sbatch master_task_a_grasp_lift_train.sh
```

Override knobs (examples):

```bash
cd ~/yash/leap_isaac_integration
NUM_ENVS=1024 VIDEO_INTERVAL=2000 VIDEO_LENGTH=300 WANDB_PROJECT=rrc_grasp_lift sbatch master_task_a_grasp_lift_train.sh
```

### Interactive training (compute node only)

```bash
cd ~/yash/leap_isaac_integration
bash task_a_grasp_lift_train.sh
```

### Where outputs go

- RL-Games logs:
  - `LEAP_Hand_Isaac_Lab/logs/rl_games/leap_hand_grasp_lift/<run_timestamp>/`
- Videos:
  - `.../videos/train/*.mp4`
- Checkpoints:
  - `.../nn/*.pth`

### Resume boilerplate (how to add later)

Your reorient pipeline already has a good resume pattern (`resume_train_leap.sh`).
For Task A, copy that pattern and switch:

- task id → `Isaac-Leap-Grasp-Lift-v0`
- nn dir prefix → `leap_hand_grasp_lift`
- train script → `LEAP_Hand_Isaac_Lab/grasp_lift_task/train_rl_games.py`

If you want, I can add the Task A resume scripts too (mirroring your existing checkpoint selection logic).

---

## “How to make more training types” boilerplate template

When you create a new task folder (e.g. `arm_reach_task/`), keep these three layers separate:

### A) Env config (`*_env_cfg.py`)

Boilerplate:

```python
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils import configclass

@configclass
class MyEnvCfg(DirectRLEnvCfg):
    action_space = ...
    observation_space = ...
    scene = ...
    sim = ...
    robot_cfg = ...
    object_cfg = ...
```

### B) Env logic (`*_env.py`)

Boilerplate methods you always implement:

```python
class MyEnv(DirectRLEnv):
    def _setup_scene(self): ...
    def _pre_physics_step(self, actions): ...
    def _apply_action(self): ...
    def _get_observations(self): return {"policy": obs}
    def _get_rewards(self): return rew
    def _get_dones(self): return terminated, time_out
    def _reset_idx(self, env_ids): ...
```

### C) Training config (RL-Games)

Boilerplate changes in YAML:

- `config.name`: unique run name
- `horizon_length`, `minibatch_size`, `mini_epochs`: sized around your `num_envs`

