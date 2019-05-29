nncase
=========================================
[![Build status](https://ci.appveyor.com/api/projects/status/cybsf4av9e2ms447/branch/master?svg=true)](https://ci.appveyor.com/project/sunnycase/nncase/branch/master)

`nncase` is a cross-platform neural network optimization toolkit for fast inference.

## Usage
Download prebuilt binaries from [Release](https://github.com/kendryte/nncase/releases).

`ncc -i <input format> -o <output format> [--dataset <dataset path>] [--postprocess <dataset postprocess>] [--weights-bits <weights quantization bits>] <input path> <output path>`

- `-i` Input format

| value | description |
|-------|------------------ |
|tflite|`.tflite` TFLite model
|paddle|`__model__` PaddlePaddle model
|caffe|`.caffemodel` Caffe model
|k210model|`.kmodel` K210 model (Only supported in inference mode)

- `-o` Output format

| value | description |
|-------|------------------ |
|k210model|`.kmodel` K210 model
|tf|`.pb` TensorFlow model
|tflite|`.tflite` TFLite model
|inference|`.bin` Model's raw output (Only support k210model input)

- `--inference-type` Inference type

| value | description |
|-------|------------------ |
|uint8| Use quantized kernels (default)
|float| Use float kernels

- `--dataset` Dataset path, **required** when the output format is `inference` or `k210model` with inference type equals `uint8`.

- `--postprocess` Dataset postprocess method

| value | description |
|-------|------------------ |
|0to1|normalize images to [0, 1]
|n1to1|normalize images to [-1, 1]

- `--weights-bits` Weights quantization bits

| value | description |
|-------|------------------ |
|8|8bit quantization [0, 255]
|16|16bit quantization [0, 65535]

- `--float-fc` Use float fullyconnected kernels.

- `--channelwise-output` Use channelwise quantization for output layers.

## Examples
- Convert TFLite model to K210 model.

  `ncc -i tflite -o k210model --dataset ./images ./mbnetv1.tflite ./mbnetv1.kmodel`

- Convert PaddlePaddle model to TensorFlow model.

  `ncc -i paddle -o tf ./MobileNetV1_pretrained ./mbnetv1.pb`

- Inference K210 model and get output binaries.

  `ncc -i k210model -o inference --dataset ./images ./mbnetv1.kmodel ./output`

- Tutorials
  - 20 classes object detection
  - Iris flowers classification

  See https://github.com/kendryte/nncase/tree/master/examples

## Supported layers

| layer | parameters |
|-------|------------------ |
| Conv2d | kernel={3x3,1x1} stride={1,2} padding=same *|
| DepthwiseConv2d | kernel={3x3,1x1} stride={1,2} padding=same *|
| FullyConnected | |
| Add | |
| MaxPool2d | |
| AveragePool2d | |
| GlobalAveragePool2d | |
| BatchNormalization | |
| BiasAdd | |
| Relu | |
| Relu6 | |
| LeakyRelu | |
| Concatenation | |
| L2Normalization | |
| Sigmoid | |
| Softmax | |
| Flatten | |
| ResizeNearestNeighbor | |

\* When using TensorFlow Conv2d/DepthwiseConv2d kernel=3x3 stride=2 padding=same, you must first use tf.pad([[0,0],[1,1],[1,1],[0,0]]) to pad the input and then use Conv2d/DepthwiseConv2d with `valid` padding.