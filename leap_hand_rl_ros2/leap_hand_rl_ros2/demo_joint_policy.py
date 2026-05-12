"""Publish demo sinusoidal joint targets on /hand/joint_commands (no trained network)."""

import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class DemoJointPolicy(Node):
    def __init__(self) -> None:
        super().__init__("demo_joint_policy")
        self.declare_parameter("num_joints", 16)
        self.declare_parameter("amplitude", 0.35)
        self.declare_parameter("rate_hz", 30.0)
        self._n = int(self.get_parameter("num_joints").value)
        self._amp = float(self.get_parameter("amplitude").value)
        self._pub = self.create_publisher(Float64MultiArray, "/hand/joint_commands", 10)
        rate = float(self.get_parameter("rate_hz").value)
        self.create_timer(1.0 / rate if rate > 0 else 0.033, self._tick)
        self._t = 0.0

    def _tick(self) -> None:
        self._t += 0.033
        msg = Float64MultiArray()
        msg.data = [self._amp * math.sin(self._t * 0.7 + i * 0.2) for i in range(self._n)]
        self._pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = DemoJointPolicy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
