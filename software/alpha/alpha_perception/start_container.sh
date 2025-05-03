#!/bin/bash

CONTAINER_NAME="ros2_jazzy_dev"
IMAGE_NAME="ros2-jazzy-tensorrt"

# Check if container is running
if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
    echo "Attaching to already running container: $CONTAINER_NAME"
    docker exec -it $CONTAINER_NAME bash -c "source /opt/ros/jazzy/setup.bash && if [ -f /ros2_ws/install/local_setup.bash ]; then source /ros2_ws/install/local_setup.bash; fi && exec bash"
else
    echo "Building Docker image: $IMAGE_NAME"
    docker build -t $IMAGE_NAME .

    echo "Starting new container: $CONTAINER_NAME"
    xhost +local:docker
    xhost +SI:localuser:$(whoami)

    docker run -it --rm \
        --name $CONTAINER_NAME \
        --runtime=nvidia \
        --gpus all \
        --network host \
        --privileged \
        --pid=host \
        --ipc=host \
        -e DISPLAY=$DISPLAY \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -v $HOME/.Xauthority:/root/.Xauthority \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        -v ./ros2_ws:/ros2_ws \
        $IMAGE_NAME
fi
