## Build from source
## 从源码编译

### Linux
1. Install dependencies

- gcc >= 10
```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt install gcc-10 g++-10
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 40
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 40 
```

- cmake >=3.17
- python >= 3.6
- libgtk2.0 

```bash
sudo apt install libgtk2.0-dev -y
sudo -A apt-get install -y --no-install-recommends libx11-xcb-dev libxcb-dri3-dev libxcb-icccm4-dev libxcb-image0-dev libxcb-keysyms1-dev libxcb-randr0-dev libxcb-render-util0-dev libxcb-shape0-dev libxcb-sync-dev libxcb-util-dev libxcb-xfixes0-dev libxcb-xinerama0-dev libxcb-xkb-dev xkb-data xorg-dev
```

2. Install conan and cmake

```bash
pip install conan cmake
```

3. Clone source

```bash
git clone https://github.com/kendryte/nncase.git
```

4. Build
```bash
BUILD_TYPE=Debug
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=$BUILD_TYPE
make -j8
cmake --install . --prefix ../install
```
5. Test (optional)

Install dependencies (MacOS)
```bash
pip install tensorflow==2.5.0 matplotlib pillow onnx==1.9.0 onnx-simplifier==0.3.6 onnxoptimizer==0.2.6 onnxruntime==1.8.0
pip install torch==1.9.0 torchvision==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytest
```

Install dependencies
```bash
pip install tensorflow==2.5.0 matplotlib pillow onnx==1.9.0 onnx-simplifier==0.3.6 onnxoptimizer==0.2.6 onnxruntime==1.8.0
pip install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install pytest
```

Export environment

```bash
echo "export LD_LIBRARY_PATH=\"${PWD}/install/lib:\$LD_LIBRARY_PATH\"" >> ~/.zshrc
echo "export PYTHONPATH=\"${PWD}/install/lib:${PWD}/install/python:${PWD}/tests:\${PYTHONPATH}\"" >> ~/.zshrc
source ~/.zshrc
```

Run tests

```bash
pytest tests
```

### Windows
1. Install dependencies
- Visual Studio 2019
- cmake >=3.17
- python >= 3.6

2. Install conan cmake
```cmd
pip install conan cmake
```
3. Clone source
```cmd
git clone https://github.com/kendryte/nncase.git --recursive
```
4. Build

Open Developer Command Prompt for VS 2019

```cmd
md out && cd out
cmake .. -G "Visual Studio 16 2019" -A x64 -DNNCASE_TARGET=k210 -DCMAKE_BUILD_TYPE=Release
msbuild nncase.sln
```
5. Test (optional)

Install dependencies
```cmd
pip install conan tensorflow==2.5.0 matplotlib pillow onnx==1.9.0 onnx-simplifier==0.3.6 onnxoptimizer==0.2.6 onnxruntime==1.8.0
pip install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install pytest
```
Run tests
```cmd
pytest tests
```