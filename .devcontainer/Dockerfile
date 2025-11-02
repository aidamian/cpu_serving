FROM python:3.12-slim

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        git-lfs \
        cmake \
        ninja-build \
        numactl \
        libnuma-dev \
        libopenblas-dev \
        libomp-dev \
        nodejs \
        npm \
        pkg-config \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install --system

ENV PIP_NO_CACHE_DIR=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    HF_HOME=/workspaces/.cache/huggingface \
    PYTHONUNBUFFERED=1

RUN mkdir -p /workspaces
WORKDIR /workspaces

CMD ["/bin/bash"]
