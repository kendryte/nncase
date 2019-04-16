# 20 Classes Object Detection
## Usage
1. Convert your image to c by `image2c.py`.
```bash
python img2c.py dog.bmp
```
2. Compile your tflite model to kmodel.
```bash
ncc -i tflite -o k210model --channelwise-output --dataset images model/20classes_yolo.tflite k210/kpu_20classes_example/yolo.kmodel
```
3. Compile your program and run.
```bash
cmake .. -DPROJ=kpu_20classes_example
make
python3 kflash.py -t kpu_20classes_example.bin
```
## Result
![demo](demo.png)