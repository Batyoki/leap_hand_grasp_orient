#!/usr/bin/env bash
# Record RViz + bridge stack in one command (GPU node, headless).
# Usage:
#   bash record_full_stack.sh 60 /tmp/rviz_leap_policy.mp4
# Env:
#   USE_DEMO=true|false (default: false)
#   ROS_DISTRO=humble
#   COLCON_WS=~/ros2_ws
#   DISPLAY_NUM=99
#   RES=1920x1080
#   AUTO_INSTALL_DEPS=1 (attempt sudo apt-get for xvfb/ffmpeg)
#   LIBGL_ALWAYS_INDIRECT=1

set -euo pipefail

DURATION="${1:-30}"
OUT="${2:-/tmp/rviz_leap_policy.mp4}"
DISPLAY_NUM="${DISPLAY_NUM:-99}"
RES="${RES:-1920x1080}"
USE_DEMO="${USE_DEMO:-false}"
ROS_DISTRO="${ROS_DISTRO:-humble}"
COLCON_WS="${COLCON_WS:-${HOME}/ros2_ws}"
MICRO_MAMBA_ENV="${MICRO_MAMBA_ENV:-}"
AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS:-0}"
LIBGL_ALWAYS_INDIRECT="${LIBGL_ALWAYS_INDIRECT:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${MICRO_MAMBA_ENV}" ]] && [[ -z "${__LEAP_ROS_IN_MAMBA:-}" ]]; then
  export __LEAP_ROS_IN_MAMBA=1
  exec micromamba run -n "${MICRO_MAMBA_ENV}" --no-capture-output \
    bash "${PACKAGE_ROOT}/scripts/record_full_stack.sh" "${DURATION}" "${OUT}"
fi

SETUP_ROS="$MAMBA_ROOT_PREFIX/envs/ros_gpu/setup.bash"
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

# Some ROS setups reference unset variables; disable nounset while sourcing.
set +u
: "${AMENT_TRACE_SETUP_FILES:=}"
# shellcheck source=/dev/null
source "${SETUP_ROS}"
# shellcheck source=/dev/null
source "${SETUP_WS}"
set -u

install_dep() {
  local pkg="$1"
  if [[ "${AUTO_INSTALL_DEPS}" != "1" ]]; then
    return 1
  fi
  if command -v sudo &>/dev/null; then
    sudo apt-get update -y && sudo apt-get install -y "${pkg}"
    return $?
  fi
  return 1
}

if ! command -v Xvfb &>/dev/null; then
  if ! install_dep xvfb; then
    echo "[ERROR] Install Xvfb (e.g. apt install xvfb) or set AUTO_INSTALL_DEPS=1."
    exit 1
  fi
fi
if ! command -v ffmpeg &>/dev/null; then
  if ! install_dep ffmpeg; then
    echo "[ERROR] Install ffmpeg or set AUTO_INSTALL_DEPS=1."
    exit 1
  fi
fi

Xvfb ":${DISPLAY_NUM}" -screen 0 "${RES}x24" &
XVFB_PID=$!
export DISPLAY=":${DISPLAY_NUM}"
export LIBGL_ALWAYS_INDIRECT

cleanup() {
  kill "${RVIZ_PID:-}" 2>/dev/null || true
  kill "${XVFB_PID}" 2>/dev/null || true
}
trap cleanup EXIT
sleep 1

echo "[INFO] Starting RViz stack (use_demo=${USE_DEMO})"
ros2 launch leap_hand_rl_ros2 leap_hand_rviz.launch.py \
  "use_demo:=${USE_DEMO}" "use_rviz:=true" &
RVIZ_PID=$!

# Capture display
echo "[INFO] Recording DISPLAY=${DISPLAY} for ${DURATION}s -> ${OUT}"
ffmpeg -y -f x11grab -video_size "${RES}" -i "${DISPLAY}.0" -t "${DURATION}" \
  -c:v libx264 -preset fast -pix_fmt yuv420p "${OUT}"

echo "[INFO] Wrote ${OUT}"
