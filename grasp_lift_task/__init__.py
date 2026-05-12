"""
Local task package: LEAP Hand grasp + lift.

This folder is intentionally self-contained so we don't need to modify anything
in the repo root. Importing this module registers the Gymnasium task:
`Isaac-Leap-Grasp-Lift-v0`.
"""

import gymnasium as gym

from . import agents  # noqa: F401


_task_entry = "grasp_lift_task"

gym.register(
    id="Isaac-Leap-Grasp-Lift-v0",
    entry_point=f"{_task_entry}.leap_hand_lift_env:LeapHandGraspLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_task_entry}.leap_hand_lift_env_cfg:LeapHandGraspLiftEnvCfg",
        "rl_games_cfg_entry_point": f"{_task_entry}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Leap-Grasp-Lift-Fixed-v0",
    entry_point=f"{_task_entry}.leap_hand_lift_env_fixed:LeapHandGraspLiftFixedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_task_entry}.leap_hand_lift_env_fixed_cfg:LeapHandGraspLiftFixedEnvCfg",
        "rl_games_cfg_entry_point": f"{_task_entry}.agents:rl_games_ppo_cfg.yaml",
    },
)

