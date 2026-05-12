#!/usr/bin/env python3

"""Play a single checkpoint and record a close-up video.

This script is task-agnostic for RL-Games tasks registered in Isaac Lab.
"""

import argparse
import math
import os
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch

from isaaclab.app import AppLauncher


def _safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text)


def _set_render_resolution(width: int, height: int) -> None:
    # Best-effort: only try if carb is available in this runtime.
    try:
        import carb  # noqa: WPS433

        settings = carb.settings.get_settings()
        settings.set("/app/window/width", int(width))
        settings.set("/app/window/height", int(height))
        settings.set("/app/renderer/resolution/width", int(width))
        settings.set("/app/renderer/resolution/height", int(height))
    except Exception:
        return


def _parse_vec3(value: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not value:
        return default
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        return default
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a checkpoint and record a close-up video.")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--video_seconds", type=float, default=30.0)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--video_width", type=int, default=1280)
    parser.add_argument("--video_height", type=int, default=720)
    parser.add_argument("--output_dir", type=str, default="logs/videos")
    parser.add_argument("--camera_offset", type=str, default="0.35,0.35,0.25")
    parser.add_argument("--camera_target_offset", type=str, default="0.0,0.0,0.05")
    parser.add_argument("--camera_update_interval", type=int, default=5)
    parser.add_argument("--real_time", action="store_true", default=False)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    args.enable_cameras = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    _set_render_resolution(args.video_width, args.video_height)

    # Register tasks.
    if "Grasp-Lift" in args.task or "Leap-Grasp-Lift" in args.task:
        import grasp_lift_task  # noqa: F401
    else:
        import LEAP_Isaaclab.tasks  # noqa: F401

    from rl_games.common import env_configurations, vecenv
    from rl_games.common.player import BasePlayer
    from rl_games.torch_runner import Runner

    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    # Force playback to be lightweight on GPU memory.
    env_cfg.sim.device = args.device
    env_cfg.scene.num_envs = args.num_envs
    physx_cfg = env_cfg.sim.physx
    physx_cfg.gpu_max_rigid_contact_count = 2**20
    physx_cfg.gpu_max_rigid_patch_count = 2**20
    physx_cfg.gpu_found_lost_pairs_capacity = 2**20
    physx_cfg.gpu_found_lost_aggregate_pairs_capacity = 2**20
    physx_cfg.gpu_total_aggregate_pairs_capacity = 2**20
    physx_cfg.gpu_collision_stack_size = 2**22
    physx_cfg.gpu_heap_capacity = 2**22
    physx_cfg.gpu_temp_buffer_capacity = 2**20
    agent_cfg = load_cfg_from_registry(args.task, "rl_games_cfg_entry_point")

    resume_path = retrieve_file_path(args.checkpoint)
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")

    # Build unique output folder to avoid overwrites.
    task_name = _safe_name(args.task)
    ckpt_name = _safe_name(os.path.splitext(os.path.basename(args.checkpoint))[0])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_run_dir = os.path.join(args.output_dir, f"{task_name}_{ckpt_name}_{timestamp}")
    run_dir = base_run_dir
    suffix = 1
    while os.path.exists(run_dir):
        run_dir = f"{base_run_dir}_run{suffix}"
        suffix += 1
    os.makedirs(run_dir, exist_ok=False)

    # Compute video length in steps.
    step_dt = float(env.unwrapped.step_dt)
    video_length_steps = max(1, int(args.video_seconds / step_dt))

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=run_dir,
        step_trigger=lambda step: step == 0,
        video_length=video_length_steps,
        name_prefix=f"{task_name}_{ckpt_name}",
        disable_logger=True,
    )

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    camera_offset = np.array(_parse_vec3(args.camera_offset, (0.35, 0.35, 0.25)), dtype=float)
    target_offset = np.array(_parse_vec3(args.camera_target_offset, (0.0, 0.0, 0.05)), dtype=float)

    def update_camera() -> None:
        try:
            obj_pos = env.unwrapped.object.data.root_pos_w[0].detach().cpu().numpy()
        except Exception:
            return
        target = obj_pos + target_offset
        eye = obj_pos + camera_offset
        env.unwrapped.sim.set_camera_view(tuple(eye.tolist()), tuple(target.tolist()))

    dt = env.unwrapped.step_dt
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]

    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    timestep = 0
    update_camera()
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            obs_t = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_t, is_deterministic=True)
            obs, _, dones, _ = env.step(actions)
            if len(dones) > 0 and agent.is_rnn and agent.states is not None:
                for s in agent.states:
                    s[:, dones, :] = 0.0

        timestep += 1
        if timestep % max(1, args.camera_update_interval) == 0:
            update_camera()
        if timestep >= video_length_steps:
            break

        sleep_time = dt - (time.time() - start_time)
        if args.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
