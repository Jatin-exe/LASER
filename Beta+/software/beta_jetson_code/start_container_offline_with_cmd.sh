#!/bin/bash

export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export ISAAC_ROS_WS=/mnt/ssd/workspaces/beta_laser/
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

PLATFORM="$(uname -m)"

BASE_NAME="isaac_ros_dev-$PLATFORM"
if [[ -n "$CONFIG_CONTAINER_NAME_SUFFIX" ]]; then
    BASE_NAME="$BASE_NAME-$CONFIG_CONTAINER_NAME_SUFFIX"
fi
CONTAINER_NAME="$BASE_NAME-container"

# Check if a package and launch file argument are provided
if [[ -z "$1" || -z "$2" ]]; then
    echo "Error: Package name or launch file not specified."
    echo "Usage: $0 <package_name> <launch_file> [<launch_args>...]"
    exit 1
fi

PACKAGE_NAME="$1"
LAUNCH_FILE="$2"
shift 2  # Shift the first two arguments, leaving only the optional launch arguments

# Remove any exited containers
if docker ps -a --quiet --filter status=exited --filter name="$CONTAINER_NAME" | grep -q .; then
    docker rm "$CONTAINER_NAME" > /dev/null
fi

# Re-use existing container
if docker ps -a --quiet --filter status=running --filter name="$CONTAINER_NAME" | grep -q .; then
    if [[ $# -gt 0 ]]; then
        LAUNCH_ARGS=$(printf '"%s" ' "$@")
    else
        LAUNCH_ARGS=""
    fi

    docker exec -i -t -u admin --workdir /workspaces/isaac_ros-dev "$CONTAINER_NAME" /bin/bash -i -l -c "source ~/.bashrc && source /workspaces/isaac_ros-dev/install/local_setup.bash && ros2 launch \"$PACKAGE_NAME\" \"$LAUNCH_FILE\" $LAUNCH_ARGS"
    exit 0
fi

