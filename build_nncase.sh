#!/bin/bash
export RISCV_ROOT_PATH=/home/zhangjizhao/work/AndeSight_STD_v323/toolchains/nds64le-elf-mculib-v5d

cd out
cmake  -DCMAKE_BUILD_TYPE=Release \
       -DBUILDING_RUNTIME=1 \
       -DCMAKE_TOOLCHAIN_FILE=../toolchains/k510.baremetal.toolchain.cmake \
       ..
make -j20

nncase_runtime=/home/zhangjizhao/work/k510_soft_test/lib/nncase
rm -rf ${nncase_runtime}/*
cmake --install . --prefix ${nncase_runtime}
cd .. 
