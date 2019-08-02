mkdir out && cd out
sudo update-alternatives --set gcc /usr/bin/gcc-8
cmake .. -DNNCASE_TARGET=k210 -DCMAKE_BUILD_TYPE=Release
make -j
strip bin/ncc