#!/usr/bin/env bash
# Record RViz off-screen (GPU cluster / no monitor). Requires: Xvfb, ffmpeg, ros2 workspace sourced.
# Usage:
#   source /opt/ros/humble/setup.bash   # or your ROS distro
#   source ~/ros2_ws/install/setup.bash
#   bash record_rviz_xvfb.sh 30 /path/to/out.mp4
set -euo pipefail
DURATION="${1:-30}"
OUT="${2:-rviz_capture.mp4}"
DISPLAY_NUM="${DISPLAY_NUM:-99}"
RES="${RES:-1920x1080}"
AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS:-0}"
LIBGL_ALWAYS_INDIRECT="${LIBGL_ALWAYS_INDIRECT:-1}"

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
sleep 1

cleanup() {
  kill "${XVFB_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Launch stack without starting a second RViz from this script if you already started it;
# here we only capture whatever is on DISPLAY.
echo "[INFO] Recording DISPLAY=${DISPLAY} for ${DURATION}s -> ${OUT}"
ffmpeg -y -f x11grab -video_size "${RES}" -i "${DISPLAY}.0" -t "${DURATION}" \
  -c:v libx264 -preset fast -pix_fmt yuv420p "${OUT}"
echo "[INFO] Wrote ${OUT}"
