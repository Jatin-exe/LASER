#!/bin/bash
#
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Build ROS dependency
echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
source /opt/ros/${ROS_DISTRO}/setup.bash

export XDG_RUNTIME_DIR=/tmp/runtime-$USER
mkdir -p $XDG_RUNTIME_DIR
chmod 0700 $XDG_RUNTIME_DIR

export QT_QPA_PLATFORM=xcb

source /opt/VimbaX_2023-4/cti/SetGenTLPath.sh

echo 'export XDG_RUNTIME_DIR=/tmp/runtime-$USER' >> ~/.bashrc
echo 'export QT_QPA_PLATFORM=xcb' >> ~/.bashrc
echo 'if [ -f /opt/VimbaX_2023-4/cti/Set_GenTL_Path.sh ]; then source /opt/VimbaX_2023-4/cti/Set_GenTL_Path.sh; fi' >> ~/.bashrc
echo 'export GENICAM_GENTL64_PATH=$GENICAM_GENTL64_PATH:"/opt/VimbaX_2023-4/cti/"' >> ~/.bashrc
echo 'export CYCLONEDDS_URI=file:///workspaces/isaac_ros-dev/src/isaac_ros_common/docker/middleware_profiles/cyclone_profile.xml' >> ~/.bashrc
echo 'ROS_DOMAIN_ID=69' >> ~/.bashrc
echo 'ROS_IP=192.168.1.101' >> ~/.bashrc
echo 'ROS_HOSTNAME=192.168.1.101' >> ~/.bashrc
echo 'ROS_LOCALHOST_ONLY=0' >> ~/.bashrc

# Restart udev daemon
sudo service udev restart

source ~/.bashrc
source /workspaces/isaac_ros-dev/install/local_setup.bash
$@

