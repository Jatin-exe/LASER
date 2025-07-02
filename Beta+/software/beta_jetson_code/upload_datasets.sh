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

# Check for required arguments
if [[ -z "$1" || -z "$2" ]]; then
    echo "Usage: $0 <bucket_name> <dataset_path>"
    exit 1
fi

BUCKET="$1"
DATASET_DIR="$2"

# Clean up exited container (optional)
if docker ps -a --quiet --filter status=exited --filter name="$CONTAINER_NAME" | grep -q .; then
    docker rm "$CONTAINER_NAME" > /dev/null
fi

# Execute Python script in container
if docker ps -a --quiet --filter status=running --filter name="$CONTAINER_NAME" | grep -q .; then
    docker exec -i -t -u admin --workdir /workspaces/isaac_ros-dev "$CONTAINER_NAME" \
        /bin/bash -i -l -c "source ~/.bashrc && python3 /workspaces/isaac_ros-dev/src/laser_framework/scripts/upload_data.py --bucket \"$BUCKET\" --local-dir \"$DATASET_DIR\""
    exit 0
fi

