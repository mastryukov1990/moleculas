FROM wallies/python-cuda:3.10-cuda11.6-runtime

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
# Install base utilities
RUN apt-get update

RUN apt-get install -y build-essential

RUN apt-get install -y wget

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt  $PROJECT_ROOT/

RUN pip3 install -r requirements.txt

RUN pip3 install dgl-cu116 dglgo -f https://data.dgl.ai/wheels/repo.html

RUN pip3 install jupyter jupyter_contrib_nbextensions

RUN jupyter contrib nbextension install --user && \
    jupyter nbextensions_configurator enable --user && \
    jupyter nbextension enable code_prettify/code_prettify && \
    jupyter nbextension enable codefolding/main && \
    jupyter nbextension enable ExecuteTime && \
    jupyter nbextension enable freeze/main && \
    jupyter nbextension enable toc2/main && \
    jupyter nbextension enable  execute_time/ExecuteTime && \
    python -m ipykernel.kernelspec
