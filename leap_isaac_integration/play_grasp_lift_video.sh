#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${HOME}/yash"
ISAACLAB_DIR="${BASE_DIR}/IsaacLab"
LEAP_REPO="${BASE_DIR}/LEAP_Hand_Isaac_Lab"
LEAP_EXT="${LEAP_REPO}/source/LEAP_Isaaclab"
CONDA_SH="${BASE_DIR}/miniforge3/bin/activate"
CONDA_ENV="${CONDA_ENV:-isaac_fresh}"

CHECKPOINT="${CHECKPOINT:-/export/home/kote/yash/LEAP_Hand_Isaac_Lab/logs/rl_games/leap_hand_grasp_lift/2026-05-11_18-38-37/nn/last_leap_hand_grasp_lift_ep_50_rew_-1212.8934.pth}"
OUT_DIR="${OUT_DIR:-${LEAP_REPO}/logs/videos}"
VIDEO_SECONDS="${VIDEO_SECONDS:-30}"
VIDEO_FPS="${VIDEO_FPS:-30}"
VIDEO_WIDTH="${VIDEO_WIDTH:-1280}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-720}"
CAMERA_OFFSET="${CAMERA_OFFSET:-0.35,0.35,0.25}"
CAMERA_TARGET_OFFSET="${CAMERA_TARGET_OFFSET:-0.0,0.0,0.05}"
CAMERA_UPDATE_INTERVAL="${CAMERA_UPDATE_INTERVAL:-5}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

# Ensure Isaac Sim extension binaries are discoverable by the dynamic linker.
ISAACSIM_PKG="${CONDA_PREFIX}/lib/python3.10/site-packages/isaacsim"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
if [[ -d "${ISAACSIM_PKG}/extscache" ]]; then
  while IFS= read -r -d '' bin_dir; do
    export LD_LIBRARY_PATH="${bin_dir}:${LD_LIBRARY_PATH:-}"
  done < <(find "${ISAACSIM_PKG}/extscache" -maxdepth 2 -type d -name bin -print0)
fi

python -m pip install -q -e "${LEAP_EXT}"

cd "${LEAP_REPO}"
export PYTHONPATH="${LEAP_REPO}:${LEAP_EXT}:${PYTHONPATH:-}"

"${ISAACLAB_DIR}/isaaclab.sh" -p \
  "${BASE_DIR}/leap_isaac_integration/play_checkpoint_video.py" \
  --task Isaac-Leap-Grasp-Lift-v0 \
  --checkpoint "${CHECKPOINT}" \
  --num_envs 1 \
  --video_seconds "${VIDEO_SECONDS}" \
  --video_fps "${VIDEO_FPS}" \
  --video_width "${VIDEO_WIDTH}" \
  --video_height "${VIDEO_HEIGHT}" \
  --output_dir "${OUT_DIR}" \
  --camera_offset "${CAMERA_OFFSET}" \
  --camera_target_offset "${CAMERA_TARGET_OFFSET}" \
  --camera_update_interval "${CAMERA_UPDATE_INTERVAL}" \
  --headless \
  --device cuda:0 \
  "$@"
