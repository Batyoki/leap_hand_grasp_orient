#!/usr/bin/env python3

"""
Self-contained RL-Games training launcher for `Isaac-Leap-Grasp-Lift-v0`.

Why this exists:
- The repo's built-in trainer imports `LEAP_Isaaclab.tasks` (the installed extension registry).
- This task lives in `LEAP_Hand_Isaac_Lab/grasp_lift_task/` and is registered locally, without
  modifying any root files.
"""

import argparse
import gc
import math
import os
import pickle
import random
import sys
from datetime import datetime

import gymnasium as gym

from isaaclab.app import AppLauncher


def dump_pickle(file_path: str, obj):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def ensure_unique_dir(path: str) -> str:
    if not os.path.exists(path):
        return path
    suffix = 1
    while True:
        candidate = f"{path}_run{suffix}"
        if not os.path.exists(candidate):
            return candidate
        suffix += 1


def parse_args():
    parser = argparse.ArgumentParser(description="Train LEAP grasp-lift with RL-Games.")
    parser.add_argument("--task", type=str, default="Isaac-Leap-Grasp-Lift-v0")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--video", action="store_true", default=True)
    parser.add_argument("--video_length", type=int, default=300)
    parser.add_argument("--video_interval", type=int, default=2000)
    parser.add_argument("--video_epoch_interval", type=int, default=50)
    parser.add_argument("--max_video_envs", type=int, default=8)
    parser.add_argument("--save_frequency", type=int, default=50)
    parser.add_argument("--save_best_after", type=int, default=20)
    parser.add_argument("--gc_interval", type=int, default=50)

    # WandB (off by default to avoid login prompts in batch jobs)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="leap-grasp-lift")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)

    # append Isaac Sim app args
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    return args_cli


def main():
    args = parse_args()

    # default to headless/high-throughput unless user overrides
    if args.headless is None:
        args.headless = True
    args.enable_cameras = bool(args.video)

    if args.video and args.max_video_envs > 0 and args.num_envs > args.max_video_envs:
        print(
            f"[WARN] --video enabled; capping num_envs from {args.num_envs} to {args.max_video_envs} "
            "to reduce GPU memory pressure."
        )
        args.num_envs = args.max_video_envs

    # launch app
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # local import registers the task
    import grasp_lift_task  # noqa: F401

    from rl_games.common import env_configurations, vecenv
    from rl_games.common.algo_observer import IsaacAlgoObserver
    from rl_games.torch_runner import Runner

    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    # Optional WandB logging by subclassing the observer.
    # In batch jobs, WandB without an API key will crash (it tries to prompt/login).
    if args.wandb and not os.environ.get("WANDB_API_KEY"):
        print("[WARN] --wandb was set but WANDB_API_KEY is missing; disabling WandB.")
        args.wandb = False

    class GcIsaacObserver(IsaacAlgoObserver):
        def __init__(self, gc_interval: int):
            super().__init__()
            self._gc_interval = max(int(gc_interval), 0)

        def _maybe_collect(self, epoch_num: int):
            if self._gc_interval <= 0:
                return
            if epoch_num % self._gc_interval != 0:
                return
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        def after_print_stats(self, frame, epoch_num, total_time):
            super().after_print_stats(frame, epoch_num, total_time)
            self._maybe_collect(epoch_num)

    if args.wandb:
        import wandb

        class WandbIsaacObserver(GcIsaacObserver):
            def __init__(self):
                super().__init__(args.gc_interval)
                self._wandb_inited = False

            def after_init(self, algo):
                super().after_init(algo)
                if not self._wandb_inited:
                    wandb.init(
                        project=args.wandb_project,
                        entity=args.wandb_entity,
                        name=args.wandb_name,
                        config=getattr(algo, "config", None),
                    )
                    self._wandb_inited = True

            def after_print_stats(self, frame, epoch_num, total_time):
                super().after_print_stats(frame, epoch_num, total_time)
                # Mirror tensorboard scalars to WandB (best-effort)
                log_dict = {}
                for k, v in self.direct_info.items():
                    try:
                        if hasattr(v, "item"):
                            v = v.item()
                        log_dict[k] = v
                    except Exception:
                        pass
                if log_dict:
                    wandb.log(log_dict, step=int(frame))

        observer = WandbIsaacObserver()
    else:
        observer = GcIsaacObserver(args.gc_interval)

    # Load env/agent config from the task registry kwargs.
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(args.task, "rl_games_cfg_entry_point")

    # seed handling
    if args.seed == -1:
        args.seed = random.randint(0, 10000)
    agent_cfg["params"]["seed"] = args.seed
    env_cfg.seed = args.seed

    if args.max_iterations is not None:
        agent_cfg["params"]["config"]["max_epochs"] = args.max_iterations

    agent_cfg["params"]["config"]["save_frequency"] = int(args.save_frequency)
    agent_cfg["params"]["config"]["save_best_after"] = int(args.save_best_after)

    # Ensure minibatch_size divides batch_size to avoid RL-Games assertion errors.
    horizon_length = int(agent_cfg["params"]["config"].get("horizon_length", 0) or 0)
    batch_size = int(args.num_envs) * horizon_length if horizon_length > 0 else 0
    if batch_size > 0:
        minibatch_size = int(agent_cfg["params"]["config"].get("minibatch_size", 0) or 0)
        if minibatch_size <= 0 or minibatch_size > batch_size or (batch_size % minibatch_size != 0):
            agent_cfg["params"]["config"]["minibatch_size"] = batch_size

    if args.checkpoint is not None:
        resume_path = retrieve_file_path(args.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path

    # logging dirs
    log_root_path = os.path.abspath(os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"]))
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    unique_log_dir = ensure_unique_dir(os.path.join(log_root_path, log_dir))
    log_dir = os.path.basename(unique_log_dir)
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

    os.makedirs(os.path.join(log_root_path, log_dir), exist_ok=True)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

    # create isaac environment
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)

    # video wrapper (gym RecordVideo; works with Isaac Lab rgb_array)
    if args.video:
        steps_num = int(agent_cfg["params"]["config"].get("steps_num", 0) or 0)
        if args.video_epoch_interval and steps_num > 0:
            video_interval = int(args.video_epoch_interval) * steps_num
        else:
            video_interval = int(args.video_interval)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_root_path, log_dir, "videos", "train"),
            step_trigger=lambda step: step % video_interval == 0,
            video_length=args.video_length,
            disable_logger=True,
        )

    # wrap for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register env
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    # runner
    runner = Runner(observer)
    runner.load(agent_cfg)
    runner.reset()
    runner.run({"train": True, "play": False})

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

