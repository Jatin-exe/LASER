ARG BASE_IMAGE=ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc

FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    apt-utils \
    ca-certificates \
    openssh-client \
    iptables \
    gnupg \
    software-properties-common \
    wget \
    libc6 \
    libstdc++6 \
    curl \
    bash \
    git \
    vim \
    iptables && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/debconf/* /var/log/* /tmp/* /var/tmp/*

# Docker

RUN mkdir -pm755 /etc/apt/keyrings && curl -o /etc/apt/keyrings/docker.asc -fsSL "https://download.docker.com/linux/ubuntu/gpg" && chmod a+r /etc/apt/keyrings/docker.asc && \
    mkdir -pm755 /etc/apt/sources.list.d && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(grep UBUNTU_CODENAME= /etc/os-release | cut -d= -f2 | tr -d '\"') stable" > /etc/apt/sources.list.d/docker.list && \
    apt-get update && apt-get install --no-install-recommends -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin \
    pigz \
    xz-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/debconf/* /var/log/* /tmp/* /var/tmp/*

COPY --from=docker:dind /usr/local/bin/docker-init /usr/local/bin/docker-init

# https://github.com/docker-library/docker
ADD https://raw.githubusercontent.com/docker-library/docker/master/modprobe.sh /usr/local/bin/modprobe
ADD https://raw.githubusercontent.com/docker-library/docker/master/dockerd-entrypoint.sh /usr/local/bin/
ADD https://raw.githubusercontent.com/docker-library/docker/master/docker-entrypoint.sh /usr/local/bin/
ADD https://raw.githubusercontent.com/moby/moby/master/hack/dind /usr/local/bin/dind

ADD https://raw.githubusercontent.com/koyeb/koyeb-docker-compose/refs/heads/master/koyeb-entrypoint.sh /usr/local/bin/koyeb-entrypoint.sh

RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh /usr/local/bin/docker-entrypoint.sh /usr/local/bin/dind /usr/local/bin/koyeb-entrypoint.sh

VOLUME /var/lib/docker

# Tailscale

COPY --from=docker.io/tailscale/tailscale:stable /usr/local/bin/tailscaled /workdir/tailscaled
COPY --from=docker.io/tailscale/tailscale:stable /usr/local/bin/tailscale /workdir/tailscale
RUN mkdir -p /var/run/tailscale /var/cache/tailscale /var/lib/tailscale

# Setup for LASER/D-FINE on Tenstorrent

#RUN git clone https://github.com/tenstorrent/pytorch2.0_ttnn.git
#    cd pytorch2.0_ttnn
#    pip install -e .


#RUN pip install --no-cache-dir \
#    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 && \

RUN pip install --no-cache-dir \
    faster-coco-eval \
    PyYAML \
    tensorboard \
    scipy \
    calflops \
    transformers \
    loguru \
    matplotlib \
    onnx \
    onnxsim \
    onnxruntime \
    opencv-contrib-python-headless


RUN git clone --recursive https://github.com/Jatin-exe/LASER.git && \
    cd LASER/software/alpha/alpha_training && \
    python scripts/download_models.py && \
    python scripts/download_dataset.py && \
# hacks to make it work outside of the docker container
    mv pucks_dataset dataset && \
    ln -s D-FINE/src src && \ 
    ln -s D-FINE/configs configs && \
    ln -s $(pwd)/dataset /dataset && \
    ln -s $(pwd)/workspace /workspace && \
    cp dfine_hgnetv2_n_custom.yml D-FINE/configs/dfine/custom && \
    cp dfine_hgnetv2.yml D-FINE/configs/dfine/include && \
    cp custom_detection.yml D-FINE/configs/dataset/




WORKDIR /workdir

WORKDIR /workspace

COPY start.sh /workdir/start.sh

ENTRYPOINT ["/usr/local/bin/koyeb-entrypoint.sh"]

CMD ["/workdir/start.sh"]
