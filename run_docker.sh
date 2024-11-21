#!/bin/bash

# 设置变量
IMAGE_NAME="liushiyun/nncase:latest"
MOUNT_POINT="/nncase"
CURRENT_DIR=$(pwd)

# 启动容器
docker run -it \
    -v $CURRENT_DIR:$MOUNT_POINT \
    $IMAGE_NAME