#!/usr/bin/env bash
# Record ROS2 message flow and topic/node info without RViz/display.
# Usage: bash record_ros_flow.sh 30 /tmp/ros_flow
set -euo pipefail

DURATION="${1:-30}"
OUT_DIR="${2:-/tmp/ros_flow}"
ROS_DISTRO="${ROS_DISTRO:-humble}"
COLCON_WS="${COLCON_WS:-${HOME}/ros2_ws}"
SETUP_ROS="${ROS_SETUP:-${MAMBA_ROOT_PREFIX}/envs/ros_gpu/setup.bash}"
SETUP_WS="${COLCON_WS}/install/setup.bash"

mkdir -p "${OUT_DIR}" "${OUT_DIR}/node_info" "${OUT_DIR}/topic_info"

if [[ ! -f "${SETUP_ROS}" ]]; then
  SETUP_ROS="/opt/ros/${ROS_DISTRO}/setup.bash"
fi
if [[ ! -f "${SETUP_ROS}" ]]; then
  echo "[ERROR] Missing ROS setup: ${SETUP_ROS}"
  exit 1
fi
if [[ ! -f "${SETUP_WS}" ]]; then
  echo "[ERROR] Missing workspace setup: ${SETUP_WS}"
  exit 1
fi

# Some ROS setups reference unset variables; disable nounset while sourcing.
set +u
: "${AMENT_TRACE_SETUP_FILES:=}"
# shellcheck source=/dev/null
source "${SETUP_ROS}"
# shellcheck source=/dev/null
source "${SETUP_WS}"
set -u

# Record bag for all topics.
ros2 bag record -a -o "${OUT_DIR}/bag" >/dev/null 2>&1 &
BAG_PID=$!

sleep 1

ros2 topic list -t >"${OUT_DIR}/topics.txt" 2>&1 || true
ros2 node list >"${OUT_DIR}/nodes.txt" 2>&1 || true

while read -r node; do
  [[ -z "${node}" ]] && continue
  ros2 node info "${node}" >"${OUT_DIR}/node_info/${node//\//_}.txt" 2>&1 || true
  sleep 0.05
done <"${OUT_DIR}/nodes.txt"

while read -r topic; do
  [[ -z "${topic}" ]] && continue
  if [[ "${topic}" == *" "* ]]; then
    topic_name="${topic%% *}"
  else
    topic_name="${topic}"
  fi
  ros2 topic info -v "${topic_name}" >"${OUT_DIR}/topic_info/${topic_name//\//_}.txt" 2>&1 || true
  sleep 0.05
done <"${OUT_DIR}/topics.txt"

# Wait for requested duration, then stop bag recording.
sleep "${DURATION}"
kill -INT "${BAG_PID}" 2>/dev/null || true
wait "${BAG_PID}" 2>/dev/null || true

# Build a simple DOT graph from node info.
python - << 'PYEOF'
import glob
import os
import re

out_dir = os.environ.get("OUT_DIR", ".")
node_info_dir = os.path.join(out_dir, "node_info")

dot_lines = ["digraph ros2 {", "  rankdir=LR;", "  node [fontname=Helvetica];"]

pub_edges = set()
sub_edges = set()

for path in glob.glob(os.path.join(node_info_dir, "*.txt")):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.rstrip() for line in f]
    if not lines:
        continue
    node = lines[0].strip()
    section = None
    for line in lines[1:]:
        if line.strip().endswith(":"):
            section = line.strip().lower()
            continue
        if not line.strip() or section is None:
            continue
        m = re.match(r"\s*(/[^:]+):", line)
        if not m:
            continue
        topic = m.group(1)
        if "publishers" in section:
            pub_edges.add((node, topic))
        elif "subscribers" in section:
            sub_edges.add((topic, node))

for node, topic in sorted(pub_edges):
    dot_lines.append(f'  "{node}" -> "{topic}";')
for topic, node in sorted(sub_edges):
    dot_lines.append(f'  "{topic}" -> "{node}";')

dot_lines.append("}")

with open(os.path.join(out_dir, "ros2_graph.dot"), "w", encoding="utf-8") as f:
    f.write("\n".join(dot_lines))
PYEOF

# Render DOT graph to an image when Graphviz is available.
if command -v dot >/dev/null 2>&1; then
  dot -Tpng "${OUT_DIR}/ros2_graph.dot" -o "${OUT_DIR}/ros2_graph.png" 2>/dev/null || true
  dot -Tsvg "${OUT_DIR}/ros2_graph.dot" -o "${OUT_DIR}/ros2_graph.svg" 2>/dev/null || true
else
  echo "[WARN] Graphviz 'dot' not found. Skipping ros2_graph.png/svg rendering." >&2
fi

# Provide a plain-text summary for quick inspection.
{
  echo "[ROS2] Topics:"; cat "${OUT_DIR}/topics.txt"; echo
  echo "[ROS2] Nodes:"; cat "${OUT_DIR}/nodes.txt"; echo
} >"${OUT_DIR}/summary.txt"

echo "[INFO] ROS flow artifacts saved in ${OUT_DIR}"
