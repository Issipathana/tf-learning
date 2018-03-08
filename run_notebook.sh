#!/usr/bin/env bash

THIS_FOLDER=$(basename $(pwd))


docker run \
    --rm \
    -it \
    -p 8888:8888 \
    -p 6006:6006 \
    -v $(pwd):/notebooks/${THIS_FOLDER} \
    -e "SHELL=/bin/bash" \
    -w /notebooks \
    --name tf-learning \
    tensorflow/tensorflow:1.5.0-py3 \
    /run_jupyter.sh --NotebookApp.token='' --allow-root
