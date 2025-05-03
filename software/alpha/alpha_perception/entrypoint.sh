#!/bin/bash
set -e

# Source ROS 2 environment
source /opt/ros/jazzy/setup.bash

# Source workspace overlay if it exists
if [ -f /ros2_ws/install/local_setup.bash ]; then
    source /ros2_ws/install/local_setup.bash
fi

# Check if workspace is already built
if [ ! -d "/ros2_ws/build" ] || [ ! -d "/ros2_ws/install" ]; then
    echo "No existing build found. Building workspace with colcon..."
    cd /ros2_ws
    colcon build --symlink-install --cmake-args=-DCMAKE_BUILD_TYPE=Release --parallel-workers $(nproc) # build the workspace
    source /ros2_ws/install/local_setup.bash
else
    echo "ROS 2 workspace already built. Skipping build step."
fi

# Launch an interactive shell
exec "$@"
