<div align="center">
<img src="docs/logo.png" width="400" alt="nncase" />
</div>

[![License](https://img.shields.io/badge/license-Apache%202-blue)](https://raw.githubusercontent.com/kendryte/nncase/master/LICENSE) 
[![Build Status](https://dev.azure.com/sunnycase/nncase/_apis/build/status/kendryte.nncase?branchName=master)](https://dev.azure.com/sunnycase/nncase/_build/latest?definitionId=1&branchName=master)

`nncase` is a neural network compiler for AI accelerators.

`nncase` 是一个为 AI 加速器设计的神经网络编译器。
技术交流 QQ 群：790699378

## Install
## 安装
Download prebuilt binaries from [Release](https://github.com/kendryte/nncase/releases).

下载预编译的二进制文件 [Release](https://github.com/kendryte/nncase/releases)。

---

## Architecture
## 架构

<div align="center">
<img src="docs/arch.png" alt="nncase arch" />
</div>

### Support commonly used CNN networks
### 支持常用的 CNN 网络

- MobileNetV1/V2
- YOLOV1 YOLOV3

## Features

- Supports multiple inputs and outputs and multi-branch structure
- Static memory allocation, no heap memory acquired
- Operators fusion and optimizations
- Support float and quantized uint8 inference
- Support post quantization from float model with calibration dataset
- Flat model with zero copy loading

## 功能

- 支持多输入输出网络，支持多分支结构
- 静态内存分配，不需要堆内存
- 算子合并和优化
- 支持 float 和量化 uint8 推理
- 支持训练后量化，使用浮点模型和量化校准集
- 平坦模型，支持零拷贝加载

## Usage
## 使用方法

[Usage 使用方法](USAGE.md)

[Examples 例子](./examples)

## Supported operators
## 支持的算子

| Operator | Is Supported |
|-------|------------------ |
| Add |✅|
| ArgMax |❌|
| ArgMin |❌|
| AveragePool2D |✅|
| BatchToSpaceND |❌|
| Cast |❌|
| Concatenation |✅|
| Conv2D |✅|
| DepthwiseConv2D |✅|
| Div |✅|
| Equal |❌|
| Exp |✅|
| ExpandDims |❌|
| Floor |✅|
| FullyConnected |✅|
| Gather |❌|
| Greater |❌|
| GreaterEqual |❌|
| MaxPool2D |✅|
| Mean |✅|
| Mul |✅|
| L2Normalization |✅|
| L2Pool2D |❌|
| LessEqual |❌|
| Log |✅|
| Logistic |✅|
| LogSoftmax |❌|
| Maximum |✅|
| Minimum |✅|
| Neg |✅|
| NotEqual |❌|
| Pack |❌|
| Pad |✅|
| Pow |❌|
| PRelu |❌|
| ReduceMax |✅|
| ReduceProd |❌|
| Reshape |✅|
| ResizeBilinear |✅|
| Rsqrt |✅|
| Select |❌|
| Shape |❌|
| Sin |✅|
| Slice |❌|
| Softmax |✅|
| SpaceToDepth |❌|
| SpaceToBatchND |❌|
| SparseToDense |❌|
| Split |❌|
| Sqrt |✅|
| Square |✅|
| Squeeze |❌|
| Sub |✅|
| Sum |✅|
| Tile |❌|
| TopK |❌|
| Transpose |✅|
| TransposeConv |❌|
| LogicalOr |❌|
| OneHot |❌|
| LogicalAnd |❌|
| LogicalNot |❌|
| UnPack |❌|
| ReduceMin |✅|
| FloorDiv |❌|
| ReduceAny |❌|
| ZerosLike |❌|
| Fill |❌|
| FloorMod |❌|
| Range |❌|
| ResizeNearesetNeighbor |✅|
| LeakyRelu |✅|
| MirrorPad |❌|
| Abs |✅|
| SplitV |❌|
| Unique |❌|
| Ceil |✅|
| Reverse |❌|
| AddN |❌|
| GatherND |❌|
| Cos |✅|
| Where |❌|
| Rank |❌|
| Elu |❌|
| ReverseSequence |❌|