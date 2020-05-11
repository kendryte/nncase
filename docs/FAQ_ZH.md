## 常见问题

### K210 和 KPU
1. K210 的硬件环境是什么?

    K210 有 6MB 通用 RAM 和 2MB KPU 专用 RAM。模型的输入和输出特征图存储在 2MB KPU RAM 中。权重和其他参数存储在 6MB 通用 RAM 中。

2. 哪些算子可以被 KPU 完全加速？

    下面的约束需要全部满足。
    - 特征图尺寸：输入特征图小于等于 320x240(WxH) 同时输出特征图大于等于 4x4(WxH)，通道数在 1 到 1024。
    - Same 对称 paddings (TensorFlow 在 stride=2 同时尺寸为偶数时使用非对称 paddings)。
    - 普通 Conv2D 和 DepthwiseConv2D，卷积核为 1x1 或 3x3，stride 为 1 或 2。
    - MaxPool(2x2 或 4x4) 和 AveragePool(2x2 或 4x4)。
    - 任意逐元素激活函数 (ReLU, ReLU6, LeakyRelu, Sigmoid...), KPU 不支持 PReLU。

3. 哪些算子可以被 KPU 部分加速？

    - 非对称 paddings 或 valid paddings 卷积, nncase 会在其前后添加必要的 Pad 和 Crop。
    - 普通 Conv2D 和 DepthwiseConv2D，卷积核为 1x1 或 3x3，但 stride 不是 1 或 2. nncase 会把它分解为 KPUConv2D 和一个 StridedSlice (可能还需要 Pad)。
    - MatMul, nncase 会把它替换为一个 Pad(到 4x4)+ KPUConv2D(1x1 卷积和) + Crop(到 1x1)。
    - TransposeConv2D, nncase 会把它替换为一个 SpaceToBatch + KPUConv2D + BatchToSpace。

### 编译模型
1. Fatal: Not supported tflite opcode: DEQUANTIZE

    使用浮点 tflite 模型，nncase 会做量化。

### 部署模型
1. 运行模型是我需要归一化输入吗？

    如果它是一个 uint8 输入 (通常是量化后的模型), 你不需要归一化，只需要提供 uint8 输入 (例如 RGB888 图像)。如果它是一个 float 输入，你需要做预处理。

2. 为什么我看到 “KPU allocator cannot allocate more memory”？

    如同 "K210 和 KPU" 章节所说, 模型的输入和输出特征图存储在 2MB KPU RAM 中。单层不能超过这个限制。你可以尝试减小特征图的尺寸。

3. 为什么运行模型时我看到 “Out of memory“？

    当你编译模型时，nncase 会打印运行模型时所需工作内存使用量 (working memory usage)。通常模型会被加载到 6MB RAM 中，所以总的主存使用量是 工作内存 + 模型的大小。
  