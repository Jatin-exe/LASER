FROM nvcr.io/nvidia/tensorrt:25.04-py3

SHELL ["/bin/bash", "-c"]

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV ROS_DISTRO=jazzy

# 1. Install locale
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# 2. Add required sources and GPG key
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    gnupg2 \
    lsb-release \
    && add-apt-repository universe

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 3. Install ROS 2
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-desktop \
    ros-dev-tools \
    libeigen3-dev \
    ros-jazzy-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

# 4. Source ROS setup in default shell
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc

WORKDIR /ros2_ws

# 5. Copy entrypoint script into container
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 6. Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# 7. Default command
CMD ["bash"]
