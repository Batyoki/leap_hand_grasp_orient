## Guide 2: Everything on the GPU node (headless) + save RViz video

Goal: Run the RViz pipeline **on the cluster GPU node** (no monitor), and record an `.mp4` video automatically.

You already have a working capture helper:

- `leap_hand_rl_ros2/scripts/record_rviz_xvfb.sh`

This guide shows the reliable sequence.

---

## Prerequisites on the GPU node

Install system packages (once, if missing):

```bash
sudo apt-get update
sudo apt-get install -y xvfb ffmpeg
```

Build ROS2 workspace (once):

```bash
mkdir -p ~/ros2_ws/src
ln -sfn /export/home/kote/yash/leap_hand_rl_ros2 ~/ros2_ws/src/leap_hand_rl_ros2
cd ~/ros2_ws
source $MAMBA_ROOT_PREFIX/envs/ros_gpu/setup.bash
colcon build --packages-select leap_hand_rl_ros2
```

---

## Minimal “demo” recording (no Isaac)

Terminal 1 (start recording + virtual display):

```bash
source $MAMBA_ROOT_PREFIX/envs/ros_gpu/setup.bash
source ~/ros2_ws/install/setup.bash

export DISPLAY_NUM=99
export RES=1920x1080
export DISPLAY=:${DISPLAY_NUM}

bash /export/home/kote/yash/leap_hand_rl_ros2/scripts/record_rviz_xvfb.sh 45 /tmp/rviz_leap_demo.mp4
```

Terminal 2 (run RViz + demo publisher on the same DISPLAY):

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

export DISPLAY=:99
ros2 launch leap_hand_rl_ros2 leap_hand_rviz.launch.py use_demo:=true use_rviz:=true
```

You should get `/tmp/rviz_leap_demo.mp4`.

---

## Recording RViz while Isaac is publishing joint commands

### Recommended split

- Isaac job (GPU) runs your policy/inference and publishes `/hand/joint_commands`.
- RViz job (can be same node) consumes `/hand/joint_commands` and renders.

### Step-by-step (same node)

Terminal A (start Xvfb + recording):

```bash
source $MAMBA_ROOT_PREFIX/envs/ros_gpu/setup.bash
source ~/ros2_ws/install/setup.bash

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
export DISPLAY_NUM=99
export DISPLAY=:${DISPLAY_NUM}

bash /export/home/kote/yash/leap_hand_rl_ros2/scripts/record_rviz_xvfb.sh 90 /tmp/rviz_leap_policy.mp4
```

Terminal B (launch RViz stack; **disable demo**):

```bash
source $MAMBA_ROOT_PREFIX/envs/ros_gpu/setup.bash
source ~/ros2_ws/install/setup.bash

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
export DISPLAY=:99

ros2 launch leap_hand_rl_ros2 leap_hand_rviz.launch.py use_demo:=false use_rviz:=true
```

Terminal C (publish joint commands from your pipeline):

- For testing:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
source /opt/ros/humble/setup.bash
ros2 topic pub -r 30 /hand/joint_commands std_msgs/msg/Float64MultiArray "{data: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}"
```

- For real use: publish from Isaac inference (preferred).

---

## Making the recording fully “one command”

A simple orchestration pattern is:

1) start Xvfb + ffmpeg capture
2) start `ros2 launch ...` in background
3) run your publisher
4) stop after N seconds

If you want, I can add a helper script `scripts/record_full_stack.sh` that does this in one go.

---

## Notes / common failure modes

- If RViz crashes at startup:
  - try lowering resolution: `RES=1280x720`
  - ensure OpenGL works headlessly (some clusters need `LIBGL_ALWAYS_INDIRECT=1`)
- If no motion:
  - check `ros2 topic echo /hand/joint_commands`
  - check `ros2 topic echo /joint_states`

