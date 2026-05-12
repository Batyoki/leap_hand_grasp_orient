#!/usr/bin/env bash
#SBATCH --job-name=leap_grasp_lift_train
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=leap_grasp_lift_%j.log
set -euo pipefail

# Task A (IIIT-H RRC): LEAP Hand grasp + lift training on the GPU cluster.
#
# This script follows the same conventions as `setup_and_train_leap.sh`, but targets the
# locally-defined task in:
#   LEAP_Hand_Isaac_Lab/grasp_lift_task
#
# It runs Isaac Lab through `IsaacLab/isaaclab.sh -p ...` (required on cluster nodes with GPUs).
#
# Usage (Slurm):
#   cd ~/yash/leap_isaac_integration
#   sbatch task_a_grasp_lift_train.sh
#
# Overrides:
#   NUM_ENVS=1024 VIDEO_INTERVAL=4000 VIDEO_LENGTH=240 WANDB_PROJECT=... bash task_a_grasp_lift_train.sh

BASE_DIR="${HOME}/yash"
ISAACLAB_DIR="${BASE_DIR}/IsaacLab"
LEAP_REPO="${BASE_DIR}/LEAP_Hand_Isaac_Lab"
LEAP_EXT="${LEAP_REPO}/source/LEAP_Isaaclab"
TASK_DIR="${LEAP_REPO}/grasp_lift_task"
CONDA_SH="${BASE_DIR}/miniforge3/bin/activate"
CONDA_ENV="${CONDA_ENV:-isaac_fresh}"

if [[ ! -d "${LEAP_REPO}" ]]; then
  echo "[ERROR] Missing LEAP repo at ${LEAP_REPO}"
  exit 1
fi
if [[ ! -d "${TASK_DIR}" ]]; then
  echo "[ERROR] Missing Task A folder at ${TASK_DIR}"
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

# Non-interactive Isaac Sim acceptance
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export ACCEPT_EULA=Y
export ISAACSIM_ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export TERM="xterm-256color"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

# Avoid leaking Python paths from other environments (e.g., ros_gpu micromamba).
unset PYTHONHOME
unset PYTHONUSERBASE
unset PYTHONPATH

# Ensure Isaac Sim extension binaries are discoverable by the dynamic linker.
ISAACSIM_PKG="${CONDA_PREFIX}/lib/python3.10/site-packages/isaacsim"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
if [[ -d "${ISAACSIM_PKG}/extscache" ]]; then
  while IFS= read -r -d '' bin_dir; do
    export LD_LIBRARY_PATH="${bin_dir}:${LD_LIBRARY_PATH:-}"
  done < <(find "${ISAACSIM_PKG}/extscache" -maxdepth 2 -type d -name bin -print0)
fi

echo "[INFO] Installing LEAP Isaac Lab extension in editable mode..."
python -m pip install -q -e "${LEAP_EXT}"

cd "${LEAP_REPO}"

# Important: this task is a local python package under LEAP_Hand_Isaac_Lab/
# so we add that repo root to PYTHONPATH.
export PYTHONPATH="${LEAP_REPO}:${LEAP_EXT}:${PYTHONPATH:-}"

NUM_ENVS="${NUM_ENVS:-128}"
TASK_ID="${TASK_ID:-Isaac-Leap-Grasp-Lift-v0}"
VIDEO="${VIDEO:-1}"                 # 1=record videos, 0=disable cameras/rendering
VIDEO_INTERVAL="${VIDEO_INTERVAL:-2000}"
VIDEO_EPOCH_INTERVAL="${VIDEO_EPOCH_INTERVAL:-50}"
VIDEO_LENGTH="${VIDEO_LENGTH:-300}"
MAX_VIDEO_ENVS="${MAX_VIDEO_ENVS:-8}"
GC_INTERVAL="${GC_INTERVAL:-50}"

# WandB defaults.
# On cluster jobs, interactive login prompts will crash the run if no API key is configured.
# Therefore default to disabled; enable explicitly with WANDB=1 and a configured WANDB_API_KEY.
WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-leap-grasp-lift}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_NAME="${WANDB_NAME:-}"

if [[ "${WANDB}" == "1" ]] && [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[WARN] WANDB=1 but WANDB_API_KEY is not set; disabling WandB to avoid login prompt."
  WANDB="0"
fi

echo "[INFO] Starting Task A training: ${TASK_ID}"
if [[ "${VIDEO}" == "1" ]]; then
  if [[ "${MAX_VIDEO_ENVS}" -gt 0 ]] && [[ "${NUM_ENVS}" -gt "${MAX_VIDEO_ENVS}" ]]; then
    echo "[WARN] VIDEO=1; capping num_envs from ${NUM_ENVS} to ${MAX_VIDEO_ENVS} to reduce GPU memory usage."
    NUM_ENVS="${MAX_VIDEO_ENVS}"
  fi
  echo "[INFO] num_envs=${NUM_ENVS} headless=true video=on interval=${VIDEO_INTERVAL} epochs=${VIDEO_EPOCH_INTERVAL} length=${VIDEO_LENGTH}"
else
  echo "[INFO] num_envs=${NUM_ENVS} headless=true video=off (no cameras/rendering)"
fi

"${ISAACLAB_DIR}/isaaclab.sh" -p \
  "${TASK_DIR}/train_rl_games.py" \
  --task "${TASK_ID}" \
  --headless \
  --device cuda:0 \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERATIONS:-3000}" \
  $( [[ "${VIDEO}" == "1" ]] && echo "--video" ) \
  $( [[ "${VIDEO}" == "1" ]] && echo "--video_interval" "${VIDEO_INTERVAL}" ) \
  $( [[ "${VIDEO}" == "1" ]] && echo "--video_epoch_interval" "${VIDEO_EPOCH_INTERVAL}" ) \
  $( [[ "${VIDEO}" == "1" ]] && echo "--video_length" "${VIDEO_LENGTH}" ) \
  $( [[ "${VIDEO}" == "1" ]] && echo "--max_video_envs" "${MAX_VIDEO_ENVS}" ) \
  --gc_interval "${GC_INTERVAL}" \
  --save_frequency "${SAVE_FREQUENCY:-50}" \
  --save_best_after "${SAVE_BEST_AFTER:-20}" \
  $( [[ "${WANDB}" == "1" ]] && echo "--wandb" ) \
  --wandb_project "${WANDB_PROJECT}" \
  ${WANDB_ENTITY:+--wandb_entity "${WANDB_ENTITY}"} \
  ${WANDB_NAME:+--wandb_name "${WANDB_NAME}"} \
  "$@"

