#!/usr/bin/env bash
# Launch LEAP hand RL → RViz bridge (robot_state_publisher + joint bridge + demo + RViz).
#
# One-time setup:
#   mkdir -p ~/ros2_ws/src
#   ln -sfn /export/home/kote/yash/leap_hand_rl_ros2 ~/ros2_ws/src/leap_hand_rl_ros2
#   source /opt/ros/humble/setup.bash
#   cd ~/ros2_ws && colcon build --packages-select leap_hand_rl_ros2
#
# Run (after workspace is built):
#   bash /export/home/kote/yash/leap_hand_rl_ros2/scripts/run_leap_ros_viz.sh
#
# With micromamba ROS env:
#   MICRO_MAMBA_ENV=ros_gpu bash .../run_leap_ros_viz.sh
#
# Environment:
#   ROS_DISTRO   default: humble
#   COLCON_WS    default: ~/ros2_ws
#   USE_DEMO     default: true (set false if you publish /hand/joint_commands yourself)
#   USE_RVIZ     default: true
#
# Extra CLI args are forwarded to ros2 launch (e.g. remappings).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROS_DISTRO="${ROS_DISTRO:-humble}"
COLCON_WS="${COLCON_WS:-${HOME}/ros2_ws}"
USE_DEMO="${USE_DEMO:-true}"
USE_RVIZ="${USE_RVIZ:-true}"
MICRO_MAMBA_ENV="${MICRO_MAMBA_ENV:-}"

if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
  echo "Usage: MICRO_MAMBA_ENV=ros_gpu $0 [extra ros2 launch args...]"
  exit 0
fi

if [[ -n "${MICRO_MAMBA_ENV}" ]] && [[ -z "${__LEAP_ROS_IN_MAMBA:-}" ]]; then
  export __LEAP_ROS_IN_MAMBA=1
  exec micromamba run -n "${MICRO_MAMBA_ENV}" --no-capture-output \
    bash "${PACKAGE_ROOT}/scripts/run_leap_ros_viz.sh" "$@"
fi

SETUP_ROS="/opt/ros/${ROS_DISTRO}/setup.bash"
SETUP_WS="${COLCON_WS}/install/setup.bash"

if [[ ! -f "${SETUP_ROS}" ]]; then
  echo "[ERROR] Missing ${SETUP_ROS} — install ROS ${ROS_DISTRO} or set ROS_DISTRO=jazzy etc."
  exit 1
fi

if [[ ! -f "${SETUP_WS}" ]]; then
  echo "[ERROR] Workspace not built: ${SETUP_WS}"
  echo "  ln -sfn ${PACKAGE_ROOT} ${COLCON_WS}/src/leap_hand_rl_ros2"
  echo "  source ${SETUP_ROS} && cd ${COLCON_WS} && colcon build --packages-select leap_hand_rl_ros2"
  exit 1
fi

# shellcheck source=/dev/null
source "${SETUP_ROS}"
# shellcheck source=/dev/null
source "${SETUP_WS}"

echo "[INFO] leap_hand_rl_ros2 | distro=${ROS_DISTRO} ws=${COLCON_WS} use_demo=${USE_DEMO} use_rviz=${USE_RVIZ}"
exec ros2 launch leap_hand_rl_ros2 leap_hand_rviz.launch.py \
  "use_demo:=${USE_DEMO}" "use_rviz:=${USE_RVIZ}" "$@"
