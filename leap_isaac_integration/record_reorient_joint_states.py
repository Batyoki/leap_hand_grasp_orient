#!/usr/bin/env python3

"""Record joint states from a reorient checkpoint into an .npz file."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Record reorient joint states from a checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--output_dir", type=str, default="logs/joint_states")
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--real_time", action="store_true", default=False)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    args.enable_cameras = False
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import LEAP_Isaaclab.tasks  # noqa: F401

    from rl_games.common import env_configurations, vecenv
    from rl_games.common.player import BasePlayer
    from rl_games.torch_runner import Runner

    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    task_id = "Isaac-Reorient-Cube-Leap"
    env_cfg = parse_env_cfg(task_id, device=args.device, num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(task_id, "rl_games_cfg_entry_point")

    # Reduce PhysX GPU buffers for playback to avoid large allocations.
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

    resume_path = retrieve_file_path(args.checkpoint)
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path

    env = gym.make(task_id, cfg=env_cfg)

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

    dt = float(env.unwrapped.step_dt)
    steps = max(1, int(args.seconds / dt))

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]

    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    joint_names = list(env.unwrapped.hand.joint_names)
    joint_series: list[np.ndarray] = []

    for _ in range(steps):
        start_time = time.time()
        with torch.inference_mode():
            obs_t = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_t, is_deterministic=True)
            obs, _, dones, _ = env.step(actions)
            if len(dones) > 0 and agent.is_rnn and agent.states is not None:
                for s in agent.states:
                    s[:, dones, :] = 0.0

        joint_pos = env.unwrapped.hand.data.joint_pos[0].detach().cpu().numpy()
        joint_series.append(joint_pos.copy())

        sleep_time = dt - (time.time() - start_time)
        if args.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.filename:
        out_name = args.filename
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_name = f"reorient_joint_states_{ts}.npz"
    out_path = os.path.join(args.output_dir, out_name)

    np.savez(
        out_path,
        task=task_id,
        checkpoint=resume_path,
        dt=dt,
        joint_names=np.array(joint_names, dtype=object),
        joint_pos=np.stack(joint_series, axis=0),
    )

    print(f"[INFO] Wrote joint states to: {out_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
