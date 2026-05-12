from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import saturate


if TYPE_CHECKING:
    from .leap_hand_lift_env_cfg import LeapHandGraspLiftEnvCfg


class LeapHandGraspLiftEnv(DirectRLEnv):
    """LEAP Hand dexterous grasp-and-lift of a 5 cm cube."""

    cfg: "LeapHandGraspLiftEnvCfg"

    def __init__(self, cfg: "LeapHandGraspLiftEnvCfg", render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.hand.num_joints

        # list of actuated joints
        self.actuated_dof_indices = [self.hand.joint_names.index(j) for j in self.cfg.actuated_joint_names]
        self.actuated_dof_indices.sort()

        # fingertip + palm bodies
        self.finger_bodies = [self.hand.body_names.index(n) for n in self.cfg.fingertip_body_names]
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies)
        # Palm body name can differ across LEAP hand USD variants.
        # Prefer exact match from config; otherwise fall back to a heuristic match.
        try:
            self.palm_body = self.hand.body_names.index(self.cfg.palm_body_name)
        except ValueError:
            candidates = []
            for n in self.hand.body_names:
                ln = n.lower()
                if "palm" in ln:
                    candidates.append(n)
            # If the asset doesn't use 'palm*' naming at all, fall back to common root body names.
            if len(candidates) == 0:
                for root_name in ("base", "root", "palm", "palm_lower"):
                    if root_name in self.hand.body_names:
                        chosen = root_name
                        break
                else:
                    raise ValueError(
                        f"palm_body_name='{self.cfg.palm_body_name}' not found in hand.body_names "
                        f"and no bodies contained 'palm'. Available bodies (first 25): {self.hand.body_names[:25]}"
                    )
            else:
                # Prefer palm_lower if present; otherwise take first 'palm*' candidate.
                preferred = "palm_lower"
                chosen = preferred if preferred in candidates else candidates[0]
            self.get_logger().warning(
                f"palm_body_name='{self.cfg.palm_body_name}' not found; using '{chosen}' instead."
            )
            self.palm_body = self.hand.body_names.index(chosen)

        # joint limits
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # buffers for position targets
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # object baseline height (per-env)
        self.object_start_z = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)

        # grasp + auto-lift state
        self.grasp_counter = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
        self.grasped = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.base_start_z = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.base_target_z = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)

        # Initialize extras for RL-Games logging
        if not hasattr(self, "extras") or self.extras is None:
            self.extras = {}
        if "log" not in self.extras:
            self.extras["log"] = {}

    def _setup_scene(self):
        # add hand and objects
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.table = RigidObject(self.cfg.table_cfg)

        # ground plane (below table; helps stability if object falls)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # clone and replicate envs
        self.scene.clone_environments(copy_from_source=False)

        # register assets with scene
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["table"] = self.table

        # lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions, -1.0, 1.0)
        self._update_grasp_state()

    def _update_grasp_state(self) -> None:
        self._compute_intermediate_values()
        in_contact = self._any_fingertip_contact()
        self.grasp_counter = torch.where(
            in_contact,
            self.grasp_counter + 1,
            torch.zeros_like(self.grasp_counter),
        )
        self.grasped = self.grasp_counter >= int(self.cfg.grasp_confirm_steps)

    def _apply_action(self) -> None:
        if self.cfg.action_type == "relative":
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.cfg.act_moving_average * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = saturate(
                targets,
                self.hand_dof_lower_limits[:, self.actuated_dof_indices],
                self.hand_dof_upper_limits[:, self.actuated_dof_indices],
            )
        elif self.cfg.action_type == "absolute":
            # scale [-1,1] to joint limits
            lower = self.hand_dof_lower_limits[:, self.actuated_dof_indices]
            upper = self.hand_dof_upper_limits[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = 0.5 * (self.actions + 1.0) * (upper - lower) + lower
            self.cur_targets[:, self.actuated_dof_indices] = (
                self.cfg.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
                + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            )
            self.cur_targets[:, self.actuated_dof_indices] = saturate(self.cur_targets[:, self.actuated_dof_indices], lower, upper)
        else:
            raise ValueError(f"Unsupported action type: {self.cfg.action_type}. Must be relative or absolute.")

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        self.hand.set_joint_position_target(self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices)

        if self.cfg.auto_lift_on_grasp and bool(self.grasped.any()):
            root_pos = self.hand.data.root_pos_w.clone()
            root_quat = self.hand.data.root_quat_w.clone()
            step = float(self.cfg.lift_speed) * float(self.cfg.sim.dt) * float(self.cfg.decimation)
            target_z = self.base_target_z
            new_z = torch.minimum(root_pos[:, 2] + step, target_z)
            root_pos[:, 2] = torch.where(self.grasped, new_z, root_pos[:, 2])
            self.hand.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1))

    def _compute_intermediate_values(self):
        # hand kinematics
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies] - self.scene.env_origins.unsqueeze(1)
        self.palm_pos = self.hand.data.body_pos_w[:, self.palm_body] - self.scene.env_origins
        self.hand_dof_pos = self.hand.data.joint_pos[:, self.actuated_dof_indices]
        self.hand_dof_vel = self.hand.data.joint_vel[:, self.actuated_dof_indices]

        # object state
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w

        # Defensive: avoid NaNs/Infs propagating into the policy (can break Normal std computation).
        self.fingertip_pos = torch.nan_to_num(self.fingertip_pos)
        self.palm_pos = torch.nan_to_num(self.palm_pos)
        self.hand_dof_pos = torch.nan_to_num(self.hand_dof_pos)
        self.hand_dof_vel = torch.nan_to_num(self.hand_dof_vel)
        self.object_pos = torch.nan_to_num(self.object_pos)
        self.object_rot = torch.nan_to_num(self.object_rot)
        # normalize quaternion (w,x,y,z)
        q_norm = torch.linalg.norm(self.object_rot, dim=-1, keepdim=True).clamp_min(1e-8)
        self.object_rot = self.object_rot / q_norm

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        palm_to_obj = self.object_pos - self.palm_pos
        obs = torch.cat(
            [
                self.hand_dof_pos,
                self.hand_dof_vel,
                self.object_pos,
                self.object_rot,
                palm_to_obj,
            ],
            dim=-1,
        )
        # Clamp and sanitize observations to prevent NaNs/Infs breaking network stability.
        obs = torch.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)
        obs = torch.clamp(obs, -100.0, 100.0)
        return {"policy": obs.float()}

    def _any_fingertip_contact(self) -> torch.Tensor:
        # Isaac Lab contact sensors require registering sensors in the scene config; since this task
        # is self-contained, we use a simple geometry proxy: any fingertip within a small distance
        # of the object center counts as "contact".
        tip_dists = torch.linalg.norm(self.fingertip_pos - self.object_pos.unsqueeze(1), dim=-1)  # (N, F)
        return tip_dists.min(dim=-1).values <= float(self.cfg.contact_dist_threshold)

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        # Stage 1: reach (mean fingertip distance to object center)
        tip_dists = torch.linalg.norm(self.fingertip_pos - self.object_pos.unsqueeze(1), dim=-1)  # (N, F)
        reach_rew = -self.cfg.reach_dist_scale * tip_dists.mean(dim=-1)

        # Stage 2: contact bonus
        in_contact = self._any_fingertip_contact()
        contact_rew = torch.where(in_contact, torch.full_like(reach_rew, self.cfg.contact_bonus), torch.zeros_like(reach_rew))

        # Stage 3: lift reward (height above initial)
        lift_height = torch.clamp(self.object_pos[:, 2] - self.object_start_z, min=0.0)
        lift_rew = self.cfg.lift_height_scale * lift_height

        # Penalties: torque + fall
        torque = torch.nan_to_num(self.hand.data.computed_torque[:, self.actuated_dof_indices])
        torque_pen = (torque ** 2).sum(dim=-1) * self.cfg.torque_penalty_scale
        fallen = self.object_pos[:, 2] < (self.cfg.table_top_z - 0.05)
        fall_pen = torch.where(fallen, torch.full_like(reach_rew, self.cfg.fall_penalty), torch.zeros_like(reach_rew))

        rew = reach_rew + contact_rew + lift_rew - torque_pen - fall_pen  # negative penalties added correctly
        # Clamp rewards to prevent extreme outliers causing NaN in network
        rew = torch.nan_to_num(rew, nan=0.0, posinf=10.0, neginf=-10.0)
        rew = torch.clamp(rew, -10.0, 10.0)

        # Logging (RL-Games picks up these via IsaacAlgoObserver)
        self.extras["log"]["reach_dist_mean"] = tip_dists.mean().detach()
        self.extras["log"]["contact_rate"] = in_contact.float().mean().detach()
        self.extras["log"]["lift_height_mean"] = lift_height.mean().detach()
        self.extras["log"]["torque_pen_mean"] = torque_pen.mean().detach()

        return rew

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        # terminate on fall or leaving bounds
        fallen = self.object_pos[:, 2] < self.cfg.fallen_z
        out_of_bounds = (torch.abs(self.object_pos[:, 0]) > 0.5) | (torch.abs(self.object_pos[:, 1]) > 0.5)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return fallen | out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)

        # reset robot joints to default
        dof_pos = self.hand.data.default_joint_pos[env_ids].clone()
        dof_vel = self.hand.data.default_joint_vel[env_ids].clone()

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        # reset object on table
        obj_state = self.object.data.default_root_state[env_ids].clone()
        obj_state[:, 0:3] += self.scene.env_origins[env_ids]
        # ensure it's on table and centered
        obj_state[:, 0:3] = self.scene.env_origins[env_ids] + torch.tensor([0.0, 0.0, 0.045], device=self.device)
        obj_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        obj_state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(obj_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(obj_state[:, 7:], env_ids)

        # update baseline z for lift reward
        self.object_start_z[env_ids] = 0.045

        # reset grasp + lift state
        base_z = self.hand.data.root_pos_w[env_ids, 2].clone()
        self.base_start_z[env_ids] = base_z
        self.base_target_z[env_ids] = base_z + float(self.cfg.lift_height)
        self.grasp_counter[env_ids] = 0
        self.grasped[env_ids] = False

        # reset sensors
        # (no explicit sensors to reset; contact is computed from geometry)

