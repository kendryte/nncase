#! /bin/bash
mkdir -p ../build/runtime
cmake -S .. -B ../build/runtime \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_HALIDE=false \
        -DENABLE_OPENMP=false \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=true \
        -DENABLE_VULKAN_RUNTIME=false \
        -DBUILDING_RUNTIME=true \
        -DBUILD_BENCHMARK=false \
        -G "Ninja" \
        -DCMAKE_INSTALL_PREFIX:PATH=../runtime_install
cmake --build ../build/runtime --target install