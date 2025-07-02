#!/bin/bash

export CUDA_HOME=/usr/local/cuda-12.2
export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export ISAAC_ROS_WS=/mnt/ssd/workspaces/beta_laser/

cd ${ISAAC_ROS_WS}/src/isaac_ros_common && ./scripts/run_dev.sh --skip_image_build
