mkdir out && cd out
sudo update-alternatives --set gcc /usr/bin/gcc-8
cmake .. -DNNCASE_TARGET=k210 -DCMAKE_BUILD_TYPE=Release
make -j
strip bin/ncc

cmake -G Ninja . -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -B build

cmake --build build


# install k210 toolchain
# wget https://github.com/kendryte/kendryte-gnu-toolchain/releases/download/v8.2.0-20190409/kendryte-toolchain-ubuntu-amd64-8.2.0-20190409.tar.xz -O ../kendryte-toolchain.tar.xz
# tar xf ../kendryte-toolchain.tar.xz -C .. && sync && export RISCV_ROOT_PATH="$(realpath ../kendryte-toolchain)"

# install k210 sdk
# wget https://github.com/kendryte/kendryte-standalone-sdk/archive/refs/heads/develop.tar.gz -O ../k210-sdk.tar.gz
# tar -xvf ../k210-sdk.tar.gz -C .. && sync

cmake -G Ninja . -DCMAKE_BUILD_TYPE=Release -DBUILDING_RUNTIME=ON -DK210_SDK_DIR=$(realpath ../kendryte-standalone-sdk-develop) -DCMAKE_TOOLCHAIN_FILE=toolchains/k210.toolchain.cmake -B out

cmake --build out

rm out
rm build
