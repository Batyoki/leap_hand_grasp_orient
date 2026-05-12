#!/usr/bin/env bash
# Resume LEAP reorient RL-Games training from a saved checkpoint.
#
# RL-Games writes two kinds of files under nn/:
#   - leap_hand_reorient.pth — saved only when *mean train reward* beats the previous best (not “latest epoch”).
#   - last_leap_hand_reorient_ep_<N>_rew_<r>.pth — periodic snapshot every save_frequency epochs (~200).
#
# Default is to resume from the newest *_ep_* milestone so you continue near the true last epoch.
# To force the “best reward” file:   CHECKPOINT_SELECT=best bash resume_train_leap.sh
# To pick an exact file:             CKPT=/path/to/nn/last_...pth bash resume_train_leap.sh
#
# Example:
#   RUN_NAME=2026-05-09_20-00-57 bash resume_train_leap.sh
set -euo pipefail

BASE_DIR="${HOME}/yash"
ISAACLAB_DIR="${BASE_DIR}/IsaacLab"
LEAP_REPO="${BASE_DIR}/LEAP_Hand_Isaac_Lab"
LEAP_EXT="${LEAP_REPO}/source/LEAP_Isaaclab"
CONDA_SH="${BASE_DIR}/miniforge3/bin/activate"
CONDA_ENV="isaac_fresh"

# Default: your last completed run folder (override RUN_NAME to match your experiment directory name).
RUN_NAME="${RUN_NAME:-2026-05-09_20-00-57}"
NN_DIR="${LEAP_REPO}/logs/rl_games/leap_hand_reorient/${RUN_NAME}/nn"
CHECKPOINT_SELECT="${CHECKPOINT_SELECT:-milestone}" # milestone | best

resolve_checkpoint() {
  local best_fixed="${NN_DIR}/leap_hand_reorient.pth"
  if [[ "${CHECKPOINT_SELECT}" == "best" ]]; then
    echo "${best_fixed}"
    return
  fi
  local best_file="" best_ep=-1
  local f ep
  shopt -s nullglob
  for f in "${NN_DIR}"/last_*_ep_*_rew_*.pth; do
    ep=$(basename "${f}" | sed -n 's/.*_ep_\([0-9]*\)_rew_.*/\1/p')
    if [[ -z "${ep}" ]]; then
      continue
    fi
    if (( 10#$ep > best_ep )); then
      best_ep=$((10#$ep))
      best_file="${f}"
    fi
  done
  shopt -u nullglob
  if [[ -n "${best_file}" ]]; then
    echo "${best_file}"
    return
  fi
  echo "${best_fixed}"
}

if [[ -n "${CKPT:-}" ]]; then
  :
else
  CKPT="$(resolve_checkpoint)"
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "[ERROR] Checkpoint not found: ${CKPT}"
  echo "Set CKPT=..., or RUN_NAME=<run folder>, or CHECKPOINT_SELECT=best/milestone"
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export ACCEPT_EULA=Y
export ISAACSIM_ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export TERM="xterm-256color"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

ISAACSIM_PKG="${CONDA_PREFIX}/lib/python3.10/site-packages/isaacsim"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
if [[ -d "${ISAACSIM_PKG}/extscache" ]]; then
  while IFS= read -r -d '' bin_dir; do
    export LD_LIBRARY_PATH="${bin_dir}:${LD_LIBRARY_PATH:-}"
  done < <(find "${ISAACSIM_PKG}/extscache" -maxdepth 2 -type d -name bin -print0)
fi

python -m pip install -q -e "${LEAP_EXT}"

cd "${LEAP_REPO}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-2000}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"

echo "[INFO] Resuming from: ${CKPT}"
echo "[INFO] Continuing logs under run: ${RUN_NAME}"
echo "[INFO] (CHECKPOINT_SELECT=${CHECKPOINT_SELECT}; set CKPT explicitly to override)"

"${ISAACLAB_DIR}/isaaclab.sh" -p \
  "${LEAP_REPO}/scripts/rl_games/train.py" \
  --task Isaac-Reorient-Cube-Leap \
  --headless \
  --video \
  --video_interval "${VIDEO_INTERVAL}" \
  --video_length "${VIDEO_LENGTH}" \
  --checkpoint "${CKPT}" \
  "+agent.params.config.full_experiment_name=${RUN_NAME}" \
  "$@"
