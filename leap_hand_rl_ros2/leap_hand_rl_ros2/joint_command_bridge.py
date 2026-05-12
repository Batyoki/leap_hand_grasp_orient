"""Subscribe to low-dimensional joint targets and publish sensor_msgs/JointState for RViz2."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class JointCommandBridge(Node):
    def __init__(self) -> None:
        super().__init__("joint_command_bridge")
        self.declare_parameter(
            "joint_names",
            [f"a_{i}" for i in range(16)],
        )
        self.declare_parameter("publish_rate_hz", 30.0)
        self._joint_names = list(self.get_parameter("joint_names").value)
        self._latest: list[float] | None = None
        self._sub = self.create_subscription(
            Float64MultiArray,
            "/hand/joint_commands",
            self._on_cmd,
            10,
        )
        rate = float(self.get_parameter("publish_rate_hz").value)
        period = 1.0 / rate if rate > 0 else 0.033
        self._pub = self.create_publisher(JointState, "joint_states", 10)
        self.create_timer(period, self._tick)
        self.get_logger().info(
            "Bridge: /hand/joint_commands -> /joint_states (%d joints)" % len(self._joint_names)
        )

    def _on_cmd(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < len(self._joint_names):
            self.get_logger().warn(
                f"Expected at least {len(self._joint_names)} values, got {len(msg.data)}"
            )
        n = min(len(self._joint_names), len(msg.data))
        self._latest = [float(msg.data[i]) for i in range(n)]

    def _tick(self) -> None:
        if self._latest is None:
            return
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = list(self._joint_names[: len(self._latest)])
        out.position = list(self._latest)
        out.velocity = []
        out.effort = []
        self._pub.publish(out)


def main() -> None:
    rclpy.init()
    node = JointCommandBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
