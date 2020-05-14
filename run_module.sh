#!/usr/bin/env bash

THIS_FOLDER=$(basename $(pwd))

echo $1


docker run \
    --rm \
    -it \
    -v $(pwd):$(pwd) \
    -e "SHELL=/bin/bash" \
    -w $(pwd) \
    --name tf-learning-exec \
    tensorflow/tensorflow:1.5.0-py3 \
    python $1
