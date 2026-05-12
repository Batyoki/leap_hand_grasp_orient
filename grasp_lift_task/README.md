## What this adds

This folder defines a new Isaac Lab direct RL task:

- `Isaac-Leap-Grasp-Lift-v0`

It is **isolated**: it does not modify any existing files in `LEAP_Hand_Isaac_Lab/`.

## Files

- `__init__.py`: registers the Gymnasium task id.
- `leap_hand_lift_env_cfg.py`: scene + MDP config (LEAP hand, table, 5cm cube, obs/action sizes).
- `leap_hand_lift_env.py`: environment logic + dense multi-stage reward.
- `agents/rl_games_ppo_cfg.yaml`: PPO config tuned for 1024 envs (A100-friendly).
- `train_rl_games.py`: headless training launcher with video + WandB.
- `play_rl_games.py`: play a checkpoint with video recording.

## Run training (RL-Games)

From `LEAP_Hand_Isaac_Lab/`:

```bash
export PYTHONPATH="$PWD:$PWD/source/LEAP_Isaaclab:${PYTHONPATH}"
python grasp_lift_task/train_rl_games.py --headless --device cuda:0 --num_envs 1024
```

Videos and logs land in:

- `LEAP_Hand_Isaac_Lab/logs/rl_games/leap_hand_grasp_lift/<run>/videos/`

## How to add more training types (arm/hand/quadruped/humanoid)

Isaac Lab tasks are “just Python packages” that register a Gym id and expose:

- an **env class** (`DirectRLEnv` or Manager-based env)
- an **env cfg** class (`DirectRLEnvCfg` / `ManagerBasedRLEnvCfg`)
- a **trainer config** (RL-Games YAML, RSL-RL python cfg, etc.)

For *multiple training types*, make one folder per task, with the same structure as this one:

```
LEAP_Hand_Isaac_Lab/<my_new_task>/
  __init__.py               # gym.register(id=..., entry_point=..., kwargs=...)
  <env>_cfg.py              # scene + MDP config (robot, objects, sensors, obs/action sizes)
  <env>.py                  # reward, dones, resets, observations, action mapping
  agents/
    rl_games_ppo_cfg.yaml   # PPO knobs + network size
  train_rl_games.py         # training launcher (headless + W&B + video)
  play_rl_games.py          # checkpoint playback
```

### Steps (repeatable)

1. **Copy this folder** and rename it (e.g. `arm_reach_task/`, `humanoid_walk_task/`).
2. In the new folder’s `__init__.py`:
   - change `id="..."` to a unique gym id (e.g. `Isaac-Arm-Reach-v0`)
   - update `entry_point` to your env class
   - update `env_cfg_entry_point` to your cfg class
3. In `<env>_cfg.py`:
   - swap the **robot asset** (`ArticulationCfg`) to your robot
   - set `scene.num_envs` and `sim.dt/decimation`
   - set `action_space` / `observation_space`
4. In `<env>.py`:
   - implement `_get_observations()` (define exactly what you want the policy to see)
   - implement `_apply_action()` (map action vector to joint PD targets / torques / etc.)
   - implement `_get_rewards()` and `_get_dones()`
   - implement `_reset_idx()` (reset robot + objects + per-episode buffers)
5. In `agents/rl_games_ppo_cfg.yaml`:
   - set `config.name` (log folder name)
   - tune `horizon_length`, `minibatch_size`, `mini_epochs`, `learning_rate`
6. Train with:

```bash
export PYTHONPATH="$PWD:$PWD/source/LEAP_Isaaclab:${PYTHONPATH}"
python <my_new_task>/train_rl_games.py --headless --device cuda:0 --num_envs 1024
```

### Rule of thumb: separating “training types”

- **Different embodiment** (arm vs humanoid): make a **new env** (new folder).
- **Same embodiment, different goal** (hand rotate vs hand lift): also make a **new env** (simpler + clean logs).
- **Same env, different curriculum**: keep env fixed, create **multiple agent YAMLs** and/or Hydra overrides.

## Play a checkpoint

```bash
export PYTHONPATH="$PWD:$PWD/source/LEAP_Isaaclab:${PYTHONPATH}"
python grasp_lift_task/play_rl_games.py --checkpoint /path/to/model.pth --headless --device cuda:0
```

