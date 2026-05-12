from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from LEAP_Isaaclab.assets import LEAP_HAND_CFG


@configclass
class LeapHandGraspLiftFixedEnvCfg(DirectRLEnvCfg):
    """Fixed-base LEAP Hand grasp-and-lift (recommended for no-arm setup)."""

    # env
    decimation = 4
    episode_length_s = 8.0
    action_space = 16
    observation_space = 16 + 16 + 3 + 4 + 3  # q, qd, obj pos, obj quat, palm->obj vector
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            # Aggressively reduce GPU buffers to avoid PhysX scene creation OOM.
            gpu_max_rigid_contact_count=2**20,
            gpu_max_rigid_patch_count=2**20,
            gpu_found_lost_pairs_capacity=2**20,
            gpu_found_lost_aggregate_pairs_capacity=2**20,
            gpu_total_aggregate_pairs_capacity=2**20,
            gpu_collision_stack_size=2**21,
            gpu_heap_capacity=2**21,
            gpu_temp_buffer_capacity=2**20,
        ),
    )

    # scene replication
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=0.75, replicate_physics=False)

    # robot: fixed base hovering above the table
    # Rotate 180 deg around X so the palm faces down toward the table.
    _rot_x_pi = (0.0, 1.0, 0.0, 0.0)
    robot_cfg: ArticulationCfg = LEAP_HAND_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=LEAP_HAND_CFG.spawn.replace(
            activate_contact_sensors=True,
            rigid_props=LEAP_HAND_CFG.spawn.rigid_props.replace(disable_gravity=True),
            articulation_props=LEAP_HAND_CFG.spawn.articulation_props.replace(fix_root_link=True),
        ),
        init_state=LEAP_HAND_CFG.init_state.replace(pos=(0.0, 0.0, 0.35), rot=_rot_x_pi),
    )

    # table: static cuboid centered at (0,0,0) with thickness 4 cm
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # 5cm cube target object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.025), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # MDP control mapping: continuous actions to PD position targets
    action_type: str = "relative"  # relative | absolute
    act_moving_average: float = 1.0 / 24.0

    # bodies
    actuated_joint_names = [
        "a_0",
        "a_1",
        "a_2",
        "a_3",
        "a_4",
        "a_5",
        "a_6",
        "a_7",
        "a_8",
        "a_9",
        "a_10",
        "a_11",
        "a_12",
        "a_13",
        "a_14",
        "a_15",
    ]
    fingertip_body_names = ["fingertip", "thumb_fingertip", "fingertip_2", "fingertip_3"]
    palm_body_name: str = "base"

    # reward scales
    reach_dist_scale: float = 2.0
    contact_bonus: float = 2.0
    contact_dist_threshold: float = 0.018
    lift_height_scale: float = 20.0
    torque_penalty_scale: float = 2.0e-4
    fall_penalty: float = 10.0

    # termination thresholds
    table_top_z: float = 0.02  # half thickness
    fallen_z: float = -0.05
    success_lift_height: float = 0.10

    # grasp confirmation + auto-lift (disabled for fixed base)
    grasp_confirm_steps: int = 5
    auto_lift_on_grasp: bool = False
    lift_height: float = 0.12
    lift_speed: float = 0.04  # m/s
