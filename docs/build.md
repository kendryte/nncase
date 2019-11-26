## Build from source
## 从源码编译

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