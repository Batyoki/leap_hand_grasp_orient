#!/bin/bash
#SBATCH --job-name=leap_reorient_resume
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=leap_reorient_resume_%j.log

set -euo pipefail

echo "--- Resuming LEAP Hand training (checkpoint + same run folder) ---"
export RUN_NAME="${RUN_NAME:-2026-05-09_20-00-57}"
export CKPT="${CKPT:-}"

"/export/home/kote/yash/leap_isaac_integration/resume_train_leap.sh"
