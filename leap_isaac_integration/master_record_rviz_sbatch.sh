#!/usr/bin/env bash
#SBATCH --job-name=leap_rviz_record
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=leap_rviz_record_%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1

BASE_DIR="${HOME}/yash"
ISAACLAB_SH="${BASE_DIR}/IsaacLab/isaaclab.sh"
INTEGRATION_DIR="${BASE_DIR}/leap_isaac_integration"
ROS_PKG_DIR="${BASE_DIR}/leap_hand_rl_ros2"
LEAP_REPO="${BASE_DIR}/LEAP_Hand_Isaac_Lab"
LEAP_EXT="${LEAP_REPO}/source/LEAP_Isaaclab"
ROS_WS="${ROS_WS:-${HOME}/ros2_ws}"
CONDA_SH="${BASE_DIR}/miniforge3/bin/activate"
CONDA_ENV="${CONDA_ENV:-isaac_fresh}"
OUT_ROOT="${OUT_ROOT:-${INTEGRATION_DIR}/rviz_runs}"
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_DIR="${OUT_ROOT}/${RUN_ID}"
JS_DIR="${RUN_DIR}/joint_states"
VID_DIR="${RUN_DIR}/videos"
FLOW_DIR="${RUN_DIR}/ros_flow"

SECONDS="${SECONDS:-30}"
BUILD_ROS="${BUILD_ROS:-1}"
RECORD_RVIZ="${RECORD_RVIZ:-0}"
ROS_DISTRO="${ROS_DISTRO:-humble}"
ROS_SETUP="${ROS_SETUP:-/export/home/kote/micromamba/envs/ros_gpu/setup.bash}"

GRASP_CHECKPOINT="${GRASP_CHECKPOINT:-}"
REORIENT_CHECKPOINT="${REORIENT_CHECKPOINT:-}"

if [[ -z "${GRASP_CHECKPOINT}" ]] || [[ -z "${REORIENT_CHECKPOINT}" ]]; then
  echo "[ERROR] Set GRASP_CHECKPOINT and REORIENT_CHECKPOINT before running."
  exit 1
fi
if [[ ! -x "${ISAACLAB_SH}" ]]; then
  echo "[ERROR] Missing Isaac Lab launcher at ${ISAACLAB_SH}"
  exit 1
fi
if [[ ! -d "${LEAP_REPO}" ]]; then
  echo "[ERROR] Missing LEAP repo at ${LEAP_REPO}"
  exit 1
fi
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[ERROR] Missing conda activate script at ${CONDA_SH}"
  exit 1
fi

mkdir -p "${JS_DIR}" "${VID_DIR}" "${FLOW_DIR}"
if ! touch "${JS_DIR}/.write_test" 2>/dev/null; then
  echo "[ERROR] Cannot write to ${JS_DIR}"
  exit 1
fi
rm -f "${JS_DIR}/.write_test"

if [[ ! -f "${ROS_SETUP}" ]]; then
  ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"
fi

if [[ "${BUILD_ROS}" == "1" ]]; then
  echo "[INFO] Building ROS workspace"
  mkdir -p "${ROS_WS}/src"
  ln -sfn "${ROS_PKG_DIR}" "${ROS_WS}/src/leap_hand_rl_ros2"
  # Some ROS setups reference unset variables; disable nounset while sourcing.
  set +u
  : "${AMENT_TRACE_SETUP_FILES:=}"
  # shellcheck source=/dev/null
  source "${ROS_SETUP}"
  set -u
  pushd "${ROS_WS}" >/dev/null
  colcon build --packages-select leap_hand_rl_ros2
  popd >/dev/null
fi

# Ensure Isaac Lab uses the correct Python (avoid system Anaconda/mamba mix).
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
unset PYTHONHOME
unset PYTHONUSERBASE
unset PYTHONPATH
export PYTHONPATH="${LEAP_REPO}:${LEAP_EXT}:${PYTHONPATH:-}"

STRICT_CUDA_CHECK="${STRICT_CUDA_CHECK:-0}"
KIT_LOG_DIR="$(ls -d "${CONDA_PREFIX}"/lib/python*/site-packages/omni/logs/Kit/Isaac-Sim/4.5 2>/dev/null | head -n 1)"
check_cuda_bad_state() {
  if [[ -z "${KIT_LOG_DIR}" ]]; then
    return 0
  fi
  local kit_log
  kit_log="$(ls -1t "${KIT_LOG_DIR}"/kit_*.log 2>/dev/null | head -n 1)"
  if [[ -n "${kit_log}" ]] && grep -q "CUDA being in bad state" "${kit_log}"; then
    if [[ "${STRICT_CUDA_CHECK}" == "1" ]]; then
      echo "[ERROR] CUDA is in a bad state on this node. Resubmit to a fresh GPU node."
      return 1
    fi
    echo "[WARN] CUDA bad-state warning detected in Kit logs. Continuing (STRICT_CUDA_CHECK=0)."
  fi
  return 0
}

# Record joint states with Isaac Lab (must use isaaclab.sh to load Isaac Sim Python)
echo "[INFO] Recording grasp-lift joint states"
if ! "${ISAACLAB_SH}" -p "${INTEGRATION_DIR}/record_grasp_lift_joint_states.py" \
  --checkpoint "${GRASP_CHECKPOINT}" \
  --seconds "${SECONDS}" \
  --output_dir "${JS_DIR}" \
  --headless \
  --device cuda:0; then
  echo "[ERROR] Grasp-lift recording failed. See Isaac Lab logs above."
  exit 1
fi
echo "[INFO] Grasp-lift recording completed"
check_cuda_bad_state || exit 1

echo "[INFO] Locating grasp-lift joint state file"
GRASP_NPZ=""
for _ in $(seq 1 10); do
  GRASP_NPZ="$(ls -1t "${JS_DIR}"/grasp_lift_joint_states_*.npz 2>/dev/null | head -n 1)"
  [[ -n "${GRASP_NPZ}" ]] && break
  sleep 1
done
if [[ -z "${GRASP_NPZ}" ]]; then
  echo "[ERROR] Grasp-lift joint state file not found in ${JS_DIR}."
  ls -la "${JS_DIR}" || true
  exit 1
fi

echo "[INFO] Recording reorient joint states"
if ! "${ISAACLAB_SH}" -p "${INTEGRATION_DIR}/record_reorient_joint_states.py" \
  --checkpoint "${REORIENT_CHECKPOINT}" \
  --seconds "${SECONDS}" \
  --output_dir "${JS_DIR}" \
  --headless \
  --device cuda:0; then
  echo "[ERROR] Reorient recording failed. See Isaac Lab logs above."
  exit 1
fi
echo "[INFO] Reorient recording completed"
check_cuda_bad_state || exit 1

echo "[INFO] Locating reorient joint state file"
REORIENT_NPZ=""
for _ in $(seq 1 10); do
  REORIENT_NPZ="$(ls -1t "${JS_DIR}"/reorient_joint_states_*.npz 2>/dev/null | head -n 1)"
  [[ -n "${REORIENT_NPZ}" ]] && break
  sleep 1
done
if [[ -z "${REORIENT_NPZ}" ]]; then
  echo "[ERROR] Reorient joint state file not found in ${JS_DIR}."
  ls -la "${JS_DIR}" || true
  exit 1
fi

# Source ROS for playback commands.
set +u
: "${AMENT_TRACE_SETUP_FILES:=}"
# shellcheck source=/dev/null
source "${ROS_SETUP}"
# shellcheck source=/dev/null
source "${ROS_WS}/install/setup.bash"
set -u

RECORD_STACK_SCRIPT="${ROS_PKG_DIR}/scripts/record_full_stack.sh"
FLOW_SCRIPT="${ROS_PKG_DIR}/scripts/record_ros_flow.sh"
if [[ ! -x "${FLOW_SCRIPT}" ]]; then
  echo "[ERROR] Missing ${FLOW_SCRIPT}"
  exit 1
fi

# Grasp-lift ROS flow capture (no display)
(
  ros2 run leap_hand_rl_ros2 replay_joint_states_grasp --file "${GRASP_NPZ}" &
  REPLAY_PID=$!
  OUT_DIR="${FLOW_DIR}/grasp_lift" bash "${FLOW_SCRIPT}" "${SECONDS}" "${FLOW_DIR}/grasp_lift"
  if [[ "${RECORD_RVIZ}" == "1" ]]; then
    bash "${RECORD_STACK_SCRIPT}" "${SECONDS}" "${VID_DIR}/grasp_lift_rviz.mp4"
  fi
  kill "${REPLAY_PID}" 2>/dev/null || true
  wait "${REPLAY_PID}" 2>/dev/null || true
)

# Reorient ROS flow capture (no display)
(
  ros2 run leap_hand_rl_ros2 replay_joint_states_reorient --file "${REORIENT_NPZ}" &
  REPLAY_PID=$!
  OUT_DIR="${FLOW_DIR}/reorient" bash "${FLOW_SCRIPT}" "${SECONDS}" "${FLOW_DIR}/reorient"
  if [[ "${RECORD_RVIZ}" == "1" ]]; then
    bash "${RECORD_STACK_SCRIPT}" "${SECONDS}" "${VID_DIR}/reorient_rviz.mp4"
  fi
  kill "${REPLAY_PID}" 2>/dev/null || true
  wait "${REPLAY_PID}" 2>/dev/null || true
)

echo "[INFO] Outputs saved under ${RUN_DIR}"
