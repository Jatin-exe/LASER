#!/bin/bash

export CUDA_HOME=/usr/local/cuda-12.2
export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export ISAAC_ROS_WS=/mnt/ssd/workspaces/beta_laser/

# Print a starting message
echo "Starting the scripts in separate GNOME terminals..."

# Open the first script in a new GNOME terminal
gnome-terminal -- bash -c "source ~/.bashrc && echo 'ISAAC_ROS_WS: $ISAAC_ROS_WS' && bash ~/start_container_offline.sh; exec bash"

sleep 10

# Open the second script in a new GNOME terminal
gnome-terminal -- bash -c "source ~/.bashrc && echo 'ISAAC_ROS_WS: $ISAAC_ROS_WS' && bash ~/start_container_offline_with_cmd.sh laser_framework jetson_clock.launch.py; exec bash"

echo "Scripts have been started in separate GNOME terminals."
