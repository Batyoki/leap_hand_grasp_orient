#!/usr/bin/env bash
set -euo pipefail

# Workspace layout expected:
#   $HOME/yash/LEAP_Hand_Isaac_Lab
#   $HOME/yash/miniforge3
BASE_DIR="${HOME}/yash"
ISAACLAB_DIR="${BASE_DIR}/IsaacLab"
LEAP_REPO="${BASE_DIR}/LEAP_Hand_Isaac_Lab"
LEAP_EXT="${LEAP_REPO}/source/LEAP_Isaaclab"
CONDA_SH="${BASE_DIR}/miniforge3/bin/activate"
CONDA_ENV="isaac_fresh"

if [[ ! -d "${LEAP_REPO}" ]]; then
  echo "[ERROR] Missing LEAP repo at ${LEAP_REPO}"
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[ERROR] Missing conda activate script at ${CONDA_SH}"
  exit 1
fi

if [[ ! -x "${ISAACLAB_DIR}/isaaclab.sh" ]]; then
  echo "[ERROR] Missing Isaac Lab launcher at ${ISAACLAB_DIR}/isaaclab.sh"
  exit 1
fi

# Avoid interactive prompts when running unattended.
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export ACCEPT_EULA=Y
export ISAACSIM_ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
# `isaaclab.sh` runs `tabs`, which fails when TERM is `dumb`.
export TERM="xterm-256color"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

# Ensure Isaac Sim extension binaries are discoverable by the dynamic linker.
# Some environments miss these paths in LD_LIBRARY_PATH by default.
ISAACSIM_PKG="${CONDA_PREFIX}/lib/python3.10/site-packages/isaacsim"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
if [[ -d "${ISAACSIM_PKG}/extscache" ]]; then
  while IFS= read -r -d '' bin_dir; do
    export LD_LIBRARY_PATH="${bin_dir}:${LD_LIBRARY_PATH:-}"
  done < <(find "${ISAACSIM_PKG}/extscache" -maxdepth 2 -type d -name bin -print0)
fi

echo "[INFO] Installing LEAP Isaac Lab extension in editable mode..."
python -m pip install -e "${LEAP_EXT}"

cd "${LEAP_REPO}"
echo "[INFO] Starting LEAP training: Isaac-Reorient-Cube-Leap"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-2000}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"

"${ISAACLAB_DIR}/isaaclab.sh" -p \
  "${LEAP_REPO}/scripts/rl_games/train.py" \
  --task Isaac-Reorient-Cube-Leap \
  --headless \
  --video \
  --video_interval "${VIDEO_INTERVAL}" \
  --video_length "${VIDEO_LENGTH}" \
  "$@"
