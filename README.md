nncase
=========================================
`nncase` is a cross-platform neural network optimization toolkit for fast inference.

## NNCase Converter
A tool to convert models between many formats.
### Usage
`ncc -i <input format> -o <output format> [--dataset <dataset path>] [--postprocess <dataset postprocess>] [--weights-bits <weights quantization bits>] <input path> <output path>`

- `-i` Input format

| value | description |
|-------|------------------ |
|tflite|`.tflite` TFLite model
|paddle|`__model__` PaddlePaddle model
|caffe|`.caffemodel` Caffe model

- `-o` Output format

| value | description |
|-------|------------------ |
|tf|`.pb` TensorFlow model
|tflite|`.tflite` TFLite model
|k210model|`.kmodel` K210 model

- `--dataset` Dataset path, *required* when the output format is `k210model`.

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

### Examples
- Convert TFLite model to K210 model.

  `ncc -i tflite -o k210model --dataset ./images ./mbnetv1.tflite ./mbnetv1.kmodel`

- Convert PaddlePaddle model to TensorFlow model.

  `ncc -i paddle -o tf ./MobileNetV1_pretrained ./mbnetv1.pb`

### Supported layers

| layer | parameters |
|-------|------------------ |
| Conv2d | kernel={3x3,1x1} stride={1,2} padding=same|
| DepthwiseConv2d | kernel={3x3,1x1} stride={1,2} padding=same|
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
| Softmax | |
