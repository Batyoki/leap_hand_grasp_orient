#!/usr/bin/env python3

"""Play a trained checkpoint for `Isaac-Leap-Grasp-Lift-v0`."""

import argparse
import math
import os
import time

import gymnasium as gym
import torch

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description="Play LEAP grasp-lift with RL-Games.")
    parser.add_argument("--task", type=str, default="Isaac-Leap-Grasp-Lift-v0")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--video", action="store_true", default=True)
    parser.add_argument("--video_length", type=int, default=240)
    parser.add_argument("--real_time", action="store_true", default=False)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    args.enable_cameras = bool(args.video)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import grasp_lift_task  # noqa: F401

    from rl_games.common import env_configurations, vecenv
    from rl_games.common.player import BasePlayer
    from rl_games.torch_runner import Runner

    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(args.task, "rl_games_cfg_entry_point")

    resume_path = retrieve_file_path(args.checkpoint)
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path

    # create env
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)

    if args.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"], "videos", "play"),
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
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

    dt = env.unwrapped.step_dt
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]

    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    timestep = 0
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
        if args.video and timestep >= args.video_length:
            break

        sleep_time = dt - (time.time() - start_time)
        if args.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

