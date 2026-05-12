from __future__ import annotations

from typing import TYPE_CHECKING

from .leap_hand_lift_env import LeapHandGraspLiftEnv

if TYPE_CHECKING:
    from .leap_hand_lift_env_fixed_cfg import LeapHandGraspLiftFixedEnvCfg


class LeapHandGraspLiftFixedEnv(LeapHandGraspLiftEnv):
    """Fixed-base LEAP Hand grasp-and-lift of a 5 cm cube."""

    cfg: "LeapHandGraspLiftFixedEnvCfg"
