#!/bin/bash
#SBATCH --job-name=leap_reorient_video
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=leap_reorient_video_%j.log

set -euo pipefail

echo "--- Recording reorient playback video ---"
"/export/home/kote/yash/leap_isaac_integration/play_reorient_video.sh"
