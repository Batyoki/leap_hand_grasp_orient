# `leap_hand_rl_ros2`

ROS 2 bridge to animate LEAP-hand joint targets (same joint names as your Isaac task: `a_0` … `a_15`) in RViz2.

## Build

```bash
mkdir -p ~/ros2_ws/src
ln -sf /export/home/kote/yash/leap_hand_rl_ros2 ~/ros2_ws/src/leap_hand_rl_ros2   # or copy
cd ~/ros2_ws
source /opt/ros/humble/setup.bash   # adjust distro
colcon build --packages-select leap_hand_rl_ros2
source install/setup.bash
```

## Run (one script)

After `colcon build`, from any directory:

```bash
bash /export/home/kote/yash/leap_hand_rl_ros2/scripts/run_leap_ros_viz.sh
```

With micromamba:

```bash
MICRO_MAMBA_ENV=ros_gpu bash /export/home/kote/yash/leap_hand_rl_ros2/scripts/run_leap_ros_viz.sh
```

Manual equivalent:

```bash
source ~/.bashrc
micromamba activate ros_gpu
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 launch leap_hand_rl_ros2 leap_hand_rviz.launch.py
```

If your environment is already activated, you can omit `micromamba run` and run `ros2 launch ...` directly.

- Set `use_demo:=false` when you supply your own publisher on `/hand/joint_commands`.
- Set `use_rviz:=false` on a headless node and only log or bag topics.

## Trained checkpoint → RViz

This package does **not** load `rl_games` checkpoints (that needs PyTorch + the same network in the Isaac env). Practical options:

1. **Inference in Isaac Lab** (recommended for exact policy): add a small script under `isaac_fresh` that runs the RL-Games player and publishes `Float64MultiArray` to ROS using `rclpy` if installed there, or write joint commands to a file/UDP and a thin ROS subscriber republishes.
2. **Replay exported traces**: record joint commands from a rollout (CSV/NPZ) and publish them in a timer node (easy to add later).

The included `demo_joint_policy` only proves the RViz + TF pipeline.

## Record RViz video (cluster, no monitor)

```bash
source ~/.bashrc && micromamba activate ros_gpu
# Terminal 1: virtual display + record
export DISPLAY=:99
Xvfb :99 -screen 0 1920x1080x24 &
bash /export/home/kote/yash/leap_hand_rl_ros2/scripts/record_rviz_xvfb.sh 45 /tmp/rviz_leap.mp4
# Terminal 2: launch (same DISPLAY)
export DISPLAY=:99
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 launch leap_hand_rl_ros2 leap_hand_rviz.launch.py use_rviz:=true use_demo:=true
```

Coordinate timing so recording overlaps with RViz running (start RViz before or extend capture time).

## URDF

`urdf/leap_hand_rviz_chain.urdf` is a **debug chain** (boxes + sequential joints). For a realistic hand, vendor the URDF/meshes from [LEAP_Hand_Sim](https://github.com/leap-hand/LEAP_Hand_Sim) and remap joint names to `a_*` in the bridge or in a small config file.
