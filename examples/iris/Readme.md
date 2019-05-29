# Iris flowers classification
## Usage
1. Download `nncase` from [Release](https://github.com/kendryte/nncase/releases) and extract `ncc-linux-x86_64.tar.xz` to `~/nncase`.
```bash
mkdir ~/nncase
tar xf ncc-linux-x86_64.tar.xz -C ~/nncase
```
32. Compile your tflite model to kmodel.
```bash
~/nncase/ncc -i tflite -o k210model --inference-type float model/iris.tflite k210/kpu_iris_example/iris.kmodel
```
4. Compile your program and run.
```bash
cmake .. -DPROJ=kpu_iris_example
make
python3 kflash.py -t kpu_iris_example.bin
```
## Result
```
16.077795, 5.607656, -2.071745,
setosa
```