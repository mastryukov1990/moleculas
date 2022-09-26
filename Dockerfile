FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# Install base utilities
RUN apt-get update

RUN apt-get install -y build-essential

RUN apt-get install -y wget

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC


RUN jupyter contrib nbextension install --user && \
    jupyter nbextensions_configurator enable --user && \
    jupyter nbextension enable code_prettify/code_prettify && \
    jupyter nbextension enable codefolding/main && \
    jupyter nbextension enable ExecuteTime && \
    jupyter nbextension enable freeze/main && \
    jupyter nbextension enable toc2/main && \
    jupyter nbextension enable  execute_time/ExecuteTime && \
    python -m ipykernel.kernelspec
