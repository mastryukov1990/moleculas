FROM ubuntu:18.04

# Install base utilities
RUN apt-get update && \
    apt-get install -y build-essentials  && \
    apt-get install -y wget &&
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH


RUN sudo apt-get --purge remove "*cublas*" "cuda*"
RUN sudo apt --purge remove "nvidia*"
RUN sudo apt autoremove
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.2.89-1_amd64.deb
RUN sudo dpkg -i cuda-repo-ubuntu1804_10.2.89-1_amd64.deb
RUN sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN sudo apt update
RUN sudo apt install -y cuda-10-2
RUN sudo reboot
RUN cat /usr/local/cuda/version.txt # Check CUDA version is 10.2


# Install python environment
RUN conda env create -f environment_gpu.yml

# Activate environment
RUN conda activate graph_transformer