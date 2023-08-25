<div align="center">
<img src="docs/logo.png" width="400" alt="nncase" />
</div>

[![GitHub repository](https://img.shields.io/badge/github-repository-blue?logo=github&style=plastic)](https://github.com/kendryte/nncase)
[![Gitee repository](https://img.shields.io/badge/gitee-repository-blue?logo=gitee&style=plastic)](https://gitee.com/kendryte/nncase)
[![GitHub release](https://img.shields.io/github/v/release/kendryte/nncase?color=brightgreen&display_name=tag&logo=github&style=plastic)](https://github.com/kendryte/nncase/releases)

`nncase` is a neural network compiler for AI accelerators.

`nncase` 是一个为 AI 加速器设计的神经网络编译器。

技术交流 QQ 群：790699378

Telegram: [nncase community](https://t.me/joinchat/PPcEPZMLaTViNDI1)

## Install from binaries

## 从二进制安装

Download prebuilt binaries from [Release](https://github.com/kendryte/nncase/releases).

下载预编译的二进制文件 [Release](https://github.com/kendryte/nncase/releases)。

## Build from source

## 从源码编译

[Build from source](./docs/build.md)

## Supported operators

## 支持的算子

- [TFLite ops](./docs/tflite_ops.md)
- [Caffe ops](./docs/caffe_ops.md)
- [ONNX ops](./docs/onnx_ops.md)

## K210/K510

- [Usage](https://github.com/kendryte/nncase/blob/release/1.0/docs/USAGE_EN.md)
- [FAQ](https://github.com/kendryte/nncase/blob/release/1.0/docs/FAQ_EN.md)
- [使用说明](https://github.com/kendryte/nncase/blob/release/1.0/docs/USAGE_ZH.md)
- [常见问题](https://github.com/kendryte/nncase/blob/release/1.0/docs/FAQ_ZH.md)
- [Example](https://github.com/kendryte/nncase/blob/release/1.0/examples/user_guide/)

## K230

- [Usage](./docs/USAGE_v2_EN.md)
- [FAQ](./docs/FAQ_EN.md)
- [Example](./examples/user_guide/k230_simulate-EN.ipynb)
- [使用说明](./docs/USAGE_v2.md)
- [常见问题](./docs/FAQ_ZH.md)
- [示例](./examples/user_guide/k230_simulate-ZH.ipynb)

## Resources

## 资源
### K210
- [K210_Yolo_framework](https://github.com/zhen8838/K210_Yolo_framework)
- [Shts!'s Blog (Japanese)](https://www.shtsno24.tokyo/2020/03/nncase-v020.html)

---

## Architecture

## 架构

<div align="center">
<img src="docs/arch.png" alt="nncase arch" />
</div>

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
