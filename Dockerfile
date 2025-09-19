ARG CUDA_VERSION=11.8.0
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04

ENV LANG C.UTF-8
ENV NVIDIA_DRIVER_CAPABILITIES video,compute,utility
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt -y upgrade && \
    apt-get -y install software-properties-common apt-utils && \
    add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && \
    apt-get -y install build-essential cmake unzip git wget curl tmux sysstat \
    libtcmalloc-minimal4 pkgconf autoconf libtool vim \
    python3.10 python3.10-dev python3.10-distutils python3.10-tk \
    libsm6 libxext6 libxrender1 libssl-dev libsndfile1 libgl1 &&\
    ln -s /usr/bin/python3.10 /usr/bin/python &&\
    ln -sf /usr/bin/python3.10 /usr/bin/python3 &&\
    ln -sf /usr/bin/pip3 /usr/bin/pip &&\
    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN pip3 install poetry

# Install PyTorch
RUN pip3 install torch==2.0.0 torchvision==0.15.1 --extra-index-url \
    https://download.pytorch.org/whl/cu118

# Python dependencies managed by Poetry
WORKDIR /workdir
COPY pyproject.toml /workdir/pyproject.toml
COPY src /workdir/src
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi \
    && rm -rf /root/.cache/pypoetry
RUN git config --global --add safe.directory /workdir

COPY data/dataset /workdir/dataset
COPY baseline /workdir/baseline
RUN mkdir -p /workdir/src/soccernet_calibration/models
RUN mkdir -p /workdir/data

ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV PYTHONPATH $PYTHONPATH:/workdir:/workdir/src
ENV TORCH_HOME=/workdir/data/.torch

ARG VERSION
ENV SOCCERNET_CALIBRATION_VERSION=$VERSION

WORKDIR /workdir
