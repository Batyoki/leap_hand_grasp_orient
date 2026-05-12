#!/bin/bash
#SBATCH --job-name=leap_rviz_record
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=leap_rviz_record_%j.log

set -euo pipefail

echo "--- Recording RViz video on GPU node ---"
"/export/home/kote/yash/leap_hand_rl_ros2/scripts/record_full_stack.sh" 60 /tmp/rviz_leap_policy.mp4
