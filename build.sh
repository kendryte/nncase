mkdir out && cd out
cmake .. -DNNCASE_TARGET=k210 -DCMAKE_BUILD_TYPE=Release
make -j
strip bin/ncc