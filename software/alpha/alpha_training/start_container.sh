#!/bin/bash

# Check if dataset directory exists
if [[ ! -d ./pucks_dataset ]]; then
    echo "Directory 'pucks_dataset' not found. Please make sure to dowload the dataset before starting the docker container."
    exit 1
fi

CONTAINER_NAME="DFine"
IMAGE_NAME="d-fine"

# Check if container is running
if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
    echo "Attaching to already running container: $CONTAINER_NAME"
    docker exec -it $CONTAINER_NAME bash -c "exec bash"
else
    echo "Building Docker image: $IMAGE_NAME"
    DOCKER_BUILDKIT=1 docker build \
        -t $IMAGE_NAME .

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
        -v ./pucks_dataset:/dataset \
        -v ./D-FINE:/workspace \
        -v ./custom_detection.yml:/workspace/configs/dataset/custom_detection.yml \
        -v ./dfine_hgnetv2.yml:/workspace/configs/dfine/include/dfine_hgnetv2.yml \
        -v ./dfine_hgnetv2_n_custom.yml:/workspace/configs/dfine/custom/dfine_hgnetv2_n_custom.yml \
        -v ./train.sh:/workspace/train.sh \
        -v ./tune.sh:/workspace/tune.sh \
        -v ./test_trained_model.py:/workspace/test_trained_model.py \
        -v ./test.sh:/workspace/test.sh \
        -v ./onnx_export.sh:/workspace/onnx_export.sh \
        -v ./models:/workspace/output \
        $IMAGE_NAME \
        /bin/bash
fi
