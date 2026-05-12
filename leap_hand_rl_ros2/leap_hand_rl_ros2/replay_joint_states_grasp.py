"""Replay recorded grasp-lift joint states on /hand/joint_commands."""

import argparse
import os

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class JointStateReplay(Node):
    def __init__(self, data_path: str, loop: bool, rate_hz: float | None) -> None:
        super().__init__("replay_joint_states_grasp")
        if not os.path.isfile(data_path):
            raise FileNotFoundError(data_path)
        data = np.load(data_path, allow_pickle=True)
        self._joint_pos = data["joint_pos"]
        dt = float(data.get("dt", 1.0 / 30.0))
        self._rate = rate_hz if rate_hz is not None else 1.0 / dt
        self._loop = loop
        self._idx = 0
        self._pub = self.create_publisher(Float64MultiArray, "/hand/joint_commands", 10)
        period = 1.0 / self._rate if self._rate > 0 else dt
        self.create_timer(period, self._tick)
        self.get_logger().info(f"Replaying {self._joint_pos.shape[0]} frames at {self._rate:.2f} Hz")

    def _tick(self) -> None:
        if self._idx >= self._joint_pos.shape[0]:
            if self._loop:
                self._idx = 0
            else:
                self.get_logger().info("Replay complete")
                rclpy.shutdown()
                return
        msg = Float64MultiArray()
        msg.data = [float(x) for x in self._joint_pos[self._idx].tolist()]
        self._pub.publish(msg)
        self._idx += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay recorded grasp-lift joint states.")
    parser.add_argument("--file", required=True, help="Path to .npz joint-state recording")
    parser.add_argument("--loop", action="store_true", default=False)
    parser.add_argument("--rate", type=float, default=None, help="Override replay rate (Hz)")
    args = parser.parse_args()

    rclpy.init()
    node = JointStateReplay(args.file, args.loop, args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
