#!/bin/bash
#SBATCH --job-name=leap_grasp_lift_train
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=leap_grasp_lift_%j.log

set -euo pipefail

echo "--- Launching Task A (grasp+lift) training job ---"
"/export/home/kote/yash/leap_isaac_integration/task_a_grasp_lift_train.sh"

