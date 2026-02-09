#!/bin/bash

PARENT_DIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )/..
docker run -it --rm \
  --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --name daily_stock_analysis_dev \
  -v ${PARENT_DIR}:/app \
  -p 8000:8000 \
  -p 8080:8080 \
  nvcr.io/nvidia/pytorch:26.01-py3 \
    /bin/bash