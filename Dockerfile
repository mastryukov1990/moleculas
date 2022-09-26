FROM ubuntu:20.04

# Install base utilities
RUN apt-get update

RUN apt-get install -y build-essential

RUN apt-get install -y wget

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH


ENV CUDA_VERSION=10.2.89 \
    CUDA_PKG_VERSION=10-2=10.2.89-1 \
    NCCL_VERSION=2.5.6 \
    CUDNN_VERSION=7.6.5.32

# BASE
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl && \
    rm -rf /var/lib/apt/lists/*

# RUNTIME AND DEVEL CUDA
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends cuda-cudart-$CUDA_PKG_VERSION cuda-compat-10-2 \
                                               cuda-libraries-$CUDA_PKG_VERSION cuda-nvtx-$CUDA_PKG_VERSION libcublas10=10.2.2.89-1 \
                                               libnccl2=$NCCL_VERSION-1+cuda10.2 \
                                               cuda-nvml-dev-$CUDA_PKG_VERSION cuda-command-line-tools-$CUDA_PKG_VERSION \
                                               cuda-libraries-dev-$CUDA_PKG_VERSION cuda-minimal-build-$CUDA_PKG_VERSION \
                                               libnccl-dev=$NCCL_VERSION-1+cuda10.2 libcublas-dev=10.2.2.89-1 && \
    ln -s cuda-10.2 /usr/local/cuda && \
    apt-mark hold libnccl2 && \
    rm -rf /var/lib/apt/lists/*

ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs

# RUNTIME AND DEVEL CUDNN7
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends libcudnn7=$CUDNN_VERSION-1+cuda10.2 libcudnn7-dev=$CUDNN_VERSION-1+cuda10.2 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=10.2 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=439,driver<441"

CMD ["bash"]