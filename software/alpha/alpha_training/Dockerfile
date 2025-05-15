FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    bzip2 \
    sudo \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/miniconda3
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -f -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

# Create 'dfine' environment and install Python
RUN conda create -n dfine python=3.11.9 -y && conda init bash

# Install required Python packages in 'dfine'
RUN conda run -n dfine pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 && \
    conda run -n dfine pip install --no-cache-dir \
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

# Copy entrypoint script and make it executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
