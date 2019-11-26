## Build from source
## 从源码编译

### Linux
1. Install dependencies
- gcc >= 8
- cmake >=3.8
- python >= 3.6

2. Install conan
```bash
pip install conan
```
3. Clone source
```bash
git clone https://github.com/kendryte/nncase.git --recursive
```
4. Build
```bash
mkdir out && cd out
cmake .. -DNNCASE_TARGET=k210 -DCMAKE_BUILD_TYPE=Release
make -j
```
5. Test (optional)

Install dependencies
```bash
pip install six==1.12 conan==1.19.2 tensorflow==2.0.0 matplotlib pillow pytest
```
Run tests
```bash
pytest tests
```

### Windows
1. Install dependencies
- Visual Studio 2019
- cmake >=3.8
- python >= 3.6

2. Install conan
```cmd
pip install conan
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
pip install six==1.12 conan==1.19.2 tensorflow==2.0.0 matplotlib pillow pytest
```
Run tests
```cmd
pytest tests
```