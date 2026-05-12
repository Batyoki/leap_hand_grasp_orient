"""robot_state_publisher + joint bridge + optional demo policy + RViz2."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg = get_package_share_directory("leap_hand_rl_ros2")
    urdf_path = os.path.join(pkg, "urdf", "leap_hand_rviz_chain.urdf")
    with open(urdf_path, encoding="utf-8") as f:
        robot_description = f.read()

    declare_use_demo = DeclareLaunchArgument(
        "use_demo",
        default_value="true",
        description="Run built-in sine demo on /hand/joint_commands",
    )
    declare_use_rviz = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Start RViz2 (disable on headless / video capture nodes)",
    )

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )
    bridge = Node(
        package="leap_hand_rl_ros2",
        executable="joint_command_bridge",
        name="joint_command_bridge",
        output="screen",
    )
    demo = Node(
        package="leap_hand_rl_ros2",
        executable="demo_joint_policy",
        name="demo_joint_policy",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_demo")),
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )

    return LaunchDescription(
        [
            declare_use_demo,
            declare_use_rviz,
            rsp,
            bridge,
            demo,
            rviz,
        ]
    )
