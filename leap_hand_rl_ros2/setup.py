from glob import glob
import os

from setuptools import setup

package_name = "leap_hand_rl_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "urdf"), glob("urdf/*.urdf")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@todo",
    description="LEAP hand RL to RViz2 bridge.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "joint_command_bridge = leap_hand_rl_ros2.joint_command_bridge:main",
            "demo_joint_policy = leap_hand_rl_ros2.demo_joint_policy:main",
            "replay_joint_states_grasp = leap_hand_rl_ros2.replay_joint_states_grasp:main",
            "replay_joint_states_reorient = leap_hand_rl_ros2.replay_joint_states_reorient:main",
        ],
    },
)
