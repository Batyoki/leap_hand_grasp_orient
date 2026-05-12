#!/bin/bash
#SBATCH --job-name=leap_grasp_lift_video
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=leap_grasp_lift_video_%j.log

set -euo pipefail

echo "--- Recording grasp-lift playback video ---"
"/export/home/kote/yash/leap_isaac_integration/play_grasp_lift_video.sh"
