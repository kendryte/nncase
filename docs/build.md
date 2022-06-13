## Build from source

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

- cmake >=3.18
- python >= 3.6

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
pip install tensorflow==2.5.0 matplotlib pillow onnx==1.9.0 onnx-simplifier==0.3.6 onnxoptimizer==0.2.6 onnxruntime==1.10.0
pip install torch==1.9.0 torchvision==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytest==6.2.5
```

Install dependencies
```bash
pip install tensorflow==2.5.0 matplotlib pillow onnx==1.9.0 onnx-simplifier==0.3.6 onnxoptimizer==0.2.6 onnxruntime==1.10.0
pip install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install pytest==6.2.5
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
- cmake >=3.18
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
pip install conan tensorflow==2.5.0 matplotlib pillow onnx==1.9.0 onnx-simplifier==0.3.6 onnxoptimizer==0.2.6 onnxruntime==1.10.0
pip install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install pytest==6.2.5
```
Run tests
```cmd
pytest tests
```

### Docker

1. Pull nncase docker image
```bash
docker pull registry.cn-hangzhou.aliyuncs.com/kendryte/nncase:latest
```

You can modify /etc/docker/daemon.json to speed up docker image.
```bash
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": ["https://182kvqe1.mirror.aliyuncs.com"]
}
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker
```

2. Clone nncase
```bash
git clone https://github.com/kendryte/nncase.git
cd nncase
```

3. Run docker
```bash
docker run -it --rm -v `pwd`:/mnt -w /mnt registry.cn-hangzhou.aliyuncs.com/kendryte/nncase:latest /bin/bash -c "/bin/bash"
```

4. Build
```bash
rm -rf build && mkdir -p build
pushd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j32
make install
popd
```

5. Test (optional)
```bash
export PYTHONPATH=`pwd`/python:`pwd`/build/lib:`pwd`/tests:$PYTHONPATH
export LD_LIBRARY_PATH=`pwd`/build/lib:$LD_LIBRARY_PATH
pytest tests/importer/onnx/basic/test_relu.py
```