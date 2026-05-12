## Guide 1: RViz visualization on your Mac (same VPN / same IP space as the GPU cluster)

Goal: Isaac/cluster publishes joint commands, your **Mac runs RViz2** and visualizes the LEAP hand via `leap_hand_rl_ros2`.

This guide assumes:

- You can reach the cluster nodes over VPN (same routed IP range)
- You can SSH to the cluster (for launching ROS publishers/bridge nodes)
- You want **RViz GUI locally** (Mac), not on the cluster

---

## Architecture (recommended)

Run **ROS graph on the Mac** (GUI side), and have the cluster publish into it using `ROS_DOMAIN_ID` + DDS discovery over VPN.

### Data flow

Cluster (GPU) → publishes: `/hand/joint_commands` (`std_msgs/Float64MultiArray`)

Mac → runs:
- `robot_state_publisher` (URDF → TF)
- `joint_command_bridge` (commands → `/joint_states`)
- `rviz2`

---

## Step 0: Decide ROS2 distro + RMW

Pick one distro across machines (recommended: **Humble**).

Pick one DDS implementation that behaves well over VPN:

- **CycloneDDS** is usually the simplest over routed networks.
- FastDDS can also work but can require extra config.

On **both Mac and cluster**, set:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42   # pick one number and keep it consistent
```

If discovery over VPN is unreliable, use CycloneDDS unicast discovery (see “VPN troubleshooting” below).

---

## Step 1: Build `leap_hand_rl_ros2` on the Mac

```bash
mkdir -p ~/ros2_ws/src
ln -sfn /export/home/kote/yash/leap_hand_rl_ros2 ~/ros2_ws/src/leap_hand_rl_ros2
cd ~/ros2_ws
source /opt/ros/humble/setup.bash   # on mac: use your ROS install path
colcon build --packages-select leap_hand_rl_ros2
source install/setup.bash
```

Sanity check that the nodes exist:

```bash
ros2 pkg executables leap_hand_rl_ros2
```

---

## Step 2: Run RViz stack on the Mac

Terminal on Mac:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch leap_hand_rl_ros2 leap_hand_rviz.launch.py use_demo:=false use_rviz:=true
```

This starts:
- `robot_state_publisher`
- `joint_command_bridge` (listens on `/hand/joint_commands`)
- RViz2

In RViz: add display **RobotModel** and **TF**.

---

## Step 3: Publish joint commands from the cluster

You have multiple options. The *simplest* for connectivity testing:

On a cluster compute node (or any node on VPN route):

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
source /opt/ros/humble/setup.bash

# Send 16 zeros at ~10 Hz:
ros2 topic pub -r 10 /hand/joint_commands std_msgs/msg/Float64MultiArray "{data: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}"
```

If RViz updates on your Mac, networking is working.

---

## Step 4 (boilerplate): Publishing from Isaac policy rollout

Recommended pattern for correctness:

- Run inference in Isaac Lab (same codepath as training).
- At each step, publish the **current 16 joint targets** (or joint positions) to `/hand/joint_commands`.

Boilerplate pseudo-code (cluster side):

```python
# inside your Isaac inference loop
msg = Float64MultiArray()
msg.data = hand_joint_targets.tolist()  # 16 floats
pub.publish(msg)
```

You can implement this either:

- in a separate ROS2 Python node that reads a socket/file and republishes, or
- directly in Isaac loop if `rclpy` is available in the same Python environment.

---

## VPN troubleshooting (DDS discovery)

ROS2 discovery is multicast-heavy by default; many VPNs block multicast.
Use CycloneDDS config with explicit peers.

On Mac, create `~/cyclonedds.xml`:

```xml
<CycloneDDS>
  <Domain id="any">
    <General>
      <AllowMulticast>false</AllowMulticast>
    </General>
    <Discovery>
      <Peers>
        <!-- put your cluster node VPN IP here -->
        <Peer address="10.0.0.123"/>
      </Peers>
    </Discovery>
  </Domain>
</CycloneDDS>
```

Then export on both sides:

```bash
export CYCLONEDDS_URI=file://$HOME/cyclonedds.xml
```

---

## Quick debug commands

On Mac:

```bash
ros2 topic list
ros2 topic echo /hand/joint_commands
ros2 topic echo /joint_states
```

If `/hand/joint_commands` arrives but `/joint_states` doesn’t, your bridge isn’t running.

