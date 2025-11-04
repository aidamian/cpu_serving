FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        git-lfs \
        lsb-release \
        ninja-build \
        numactl \
        libnuma-dev \
        libopenblas-dev \
        libomp-dev \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        wget \
        nodejs \
        npm \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install --system

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

ENV PIP_NO_CACHE_DIR=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    UV_CACHE_DIR=/root/.cache/uv \
    HF_HOME=/workspaces/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    VIRTUALENV_HOME=/opt/venvs

RUN mkdir -p /workspaces
RUN mkdir -p "$VIRTUALENV_HOME" /workspaces/.cache/huggingface

WORKDIR /workspaces

CMD ["/bin/bash"]
