# yolox多尺度推理效果体验


我们可以测试`yolox`的网络具备相当强大的多尺度预测能力,在缩小模型输入的情况下依旧可以正常识别(此脚本位于`yolox`项目中):
```bash
python tools/demo.py image -f exps/default/nano.py -c build/yolox_nano.pth --path assets/person.jpg --conf 0.3 --nms 0.65 --tsize 224 --save_result --device cpu
```

# 模型编译

## 导出onnx模型

虽然`k210`的`kpu`内存有限且摄像头采集图像大小有限,不过`yolox`的多尺度能力能最大限度的避免以上问题,我们导出输入为`224,224`的`onnx`模型也可以得到不错的精度.(此脚本位于`yolox`项目中)

```bash
python tools/export_onnx.py --output-name yolox_nano_224.onnx -f exps/default/nano.py  -c build/yolox_nano.pth --tsize 224
```

## 转换onnx到kmodel

使用[nncase](https://github.com/kendryte/nncase/tree/master)神经网络编译器对`onnx`模型进行编译优化、后训练量化得到适用于边缘计算的`kmodel`格式模型:

首先从`nncase`的`ci`页面中下载合适你架构的二进制包,然后执行(这里我们采用是`20`分类`yolo`例子中图像进行量化)

```sh
ncc compile model/yolox_nano_224.onnx k210/yolox_detect_example/yolox_nano_224.kmodel -i onnx -t k210 --dataset ../20classes_yolo/images --input-mean 0.48 --input-std 0.225
```

# PC端测试

## 测试量化性能损失

### 1. 编译浮点模型验证模型并推理得到模型结果:

```sh
ncc compile model/yolox_nano_224.onnx k210/yolox_detect_example/yolox_nano_224_float.kmodel -i onnx -t k210 --input-mean 0.48 --input-std 0.225
ncc infer k210/yolox_detect_example/yolox_nano_224_float.kmodel tmp/yolox_nano_float --dataset images --dataset-format image --input-mean 0.48 --input-std 0.225
```

### 2. 解析浮点输出并检查

```sh
python tools/decoder.py tmp/yolox_nano_float/dog.bin
```
可能的结果:
```sh
[tensor([[3.7685e+01, 5.9093e+01, 9.3770e+01, 1.5753e+02, 7.7338e-01, 8.6584e-01,
         1.6000e+01],
        [3.8933e+01, 3.3743e+01, 1.6518e+02, 1.2746e+02, 7.9930e-01, 8.2242e-01,
         1.0000e+00],
        [3.7944e+01, 5.8277e+01, 9.5360e+01, 1.5441e+02, 6.2752e-01, 5.6163e-01,
         1.5000e+01],
        [1.2648e+02, 1.4864e+01, 2.0647e+02, 4.8094e+01, 1.9365e-01, 6.6859e-01,
         2.0000e+00]])]
```

### 3. 编译定点模型验证模型并推理得到模型结果:

```sh
ncc compile model/yolox_nano_224.onnx k210/yolox_detect_example/yolox_nano_224.kmodel -i onnx -t k210 --dataset ../20classes_yolo/images --input-mean 0.48 --input-std 0.225

ncc infer k210/yolox_detect_example/yolox_nano_224.kmodel tmp/yolox_nano --dataset images --dataset-format image --input-mean 0.48 --input-std 0.225
```

### 4. 解析定点输出并检查:

这里的定点结果可能不是很好,主要原因有两个:
1.  input mean std还不支持3通道指定
2.  没有by channel量化

```sh
python tools/decoder.py tmp/yolox_nano/person.bin
```
可能的结果:
```sh
[tensor([[104.7265,  36.6313, 149.2677, 168.6473,   0.8630,   0.7738,   0.0000],
        [186.7276,  25.2410, 220.6741, 168.7648,   0.6220,   0.8149,   0.0000],
        [ -1.4380,   2.0039, 186.0058, 169.0368,   0.2888,   0.6813,  20.0000],
        [138.6658,  40.5240, 167.3869, 106.7722,   0.2294,   0.6840,   0.0000]])]
```

# K210端测试

## 生成静态图像用于测试

可以先利用我写好的脚本转换图像到`bin`文件,用于`k210`上的推理测试,以下命令将会在`yolox_detect_example`目录下生成`input.bin`用于后续测试.
```sh
python tools/make_image_bin.py images/person.jpg k210/yolox_detect_example/input.bin
```

## 浮点模型推理测试

`yolox nano`模型还是略大于k210的内存,因此无法加载.

## 定点模型推理测试

使用最新的[裸机sdk](https://github.com/kendryte/kendryte-standalone-sdk/tree/develop),将`yolox_detect_example`拷贝到`src`目录下,然后进行编译(请参考裸机sdk使用指南,首先配置好工具链等相关环境)
```bash
mkdir build && cd build
cmake .. -DPROJ=yolox_detect_example -DTOOLCHAIN=/usr/local/opt/kendryte-toolchain/bin
make -j
kflash yolox_detect_example.bin -B kd233 -p /dev/cu.usbserial-1130 -b 2000000 -t
```
注意不同的电脑上usb端口号并不一致.

可能的结果:
![demo](demo.jpg)

# 致谢
[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)