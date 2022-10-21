#!/usr/bin/env bash

docker run  --gpus all -v "WORK_DIR":/app  -p 8020:8020 --rm  -it moleculas /bin/bash -c cd / && jupyter notebook --ip=0.0.0.0 --no-browser  --allow-root  --port=8020  --NotebookApp.token=crmteam01!
