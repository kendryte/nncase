#! /bin/bash
mkdir -p ../build/simulator
cmake -S .. -B ../build/simulator \
        -DCMAKE_BUILD_TYPE=Debug \
        -DENABLE_HALIDE=false \
        -DENABLE_OPENMP=false \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=true \
        -DENABLE_VULKAN_RUNTIME=false \
        -DBUILD_BENCHMARK=false \
        -G "Ninja" \
        -DCMAKE_INSTALL_PREFIX:PATH=../simulator_install
cmake --build ../build/simulator --target install