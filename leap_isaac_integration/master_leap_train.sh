#!/bin/bash
#SBATCH --job-name=leap_reorient_train
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=leap_reorient_%j.log

set -euo pipefail

echo "--- Launching LEAP Hand training job ---"
"/export/home/kote/yash/leap_isaac_integration/setup_and_train_leap.sh"
