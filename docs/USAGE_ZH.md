## 使用方法

### 命令行
```
DESCRIPTION
NNCASE model compiler and inference tool.

SYNOPSIS
    ncc compile -i <input format> -t <target>
        <input file> [--input-prototxt <input prototxt>] <output file> [--output-arrays <output arrays>]
        [--quant-type <quant type>] [--w-quant-type <w quant type>] [--use-mse-quant-w]
        [--dataset <dataset path>] [--dataset-format <dataset format>] [--calibrate-method <calibrate method>]
        [--preprocess] [--swapRB] [--mean <normalize mean>] [--std <normalize std>]
        [--input-range <input range>] [--input-shape <input shape>] [--letterbox-value <letter box value>]
        [--input-type <input type>] [--output-type <output type>]
        [--input-layout <input layout>] [--output-layout <output layout>]
        [--is-fpga] [--dump-ir] [--dump-asm] [--dump-quant-error] [--dump-dir <dump directory>] [--benchmark-only]

    ncc infer <input file> <output path>
        --dataset <dataset path> [--dataset-format <dataset format>]
        [--input-layout <input layout>]

    ncc [-v]

OPTIONS
  compile

  -i, --input-format <input format>
                          input format, e.g. tflite|onnx|caffe
  -t, --target <target>   target architecture, e.g. cpu|k210|k510
  <input file>            input file
  --input-prototxt <input prototxt>
                          input prototxt
  <output file>           output file
  --output-arrays <output arrays>
                          output arrays
  --quant-type <quant type>
                          post trainning quantize type, e.g uint8|int8, default is uint8
  --w-quant-type <w quant type>
                          post trainning weights quantize type, e.g uint8|int8, default is uint8
  --use-mse-quant-w       use min mse algorithm to refine weights quantilization or not, default is 0
  --dataset <dataset path>
                          calibration dataset, used in post quantization
  --dataset-format <dataset format>
                          datset format: e.g. image|raw, default is image
  --calibrate-method <calibrate method>
                          calibrate method: e.g. no_clip|l2|kld_m0|kld_m1|kld_m2|cdf, default is no_clip
  --preprocess            enable preprocess, default is 0
  --swapRB                swap red and blue channel, default is 0
  --mean <normalize mean> normalize mean, default is 0.000000
  --std <normalize std>   normalize std, default is 1.000000
  --input-range <input range>
                          float range after preprocess
  --input-shape <input shape>
                          shape for input data
  --letterbox-value <letter box value>
                          letter box pad value, default is 0.000000
  --input-type <input type>
                          input type, e.g float32|uint8|default, default is default
  --output-type <output type>
                          output type, e.g float32|uint8, default is float32
  --input-layout <input layout>
                          input layout, e.g NCHW|NHWC, default is NCHW
  --output-layout <output layout>
                          output layout, e.g NCHW|NHWC, default is NCHW
  --is-fpga               use fpga parameters, default is 0
  --dump-ir               dump ir to .dot, default is 0
  --dump-asm              dump assembly, default is 0
  --dump-quant-error      dump quant error, default is 0
  --dump-dir <dump directory>
                          dump to directory
  --benchmark-only        compile kmodel only for benchmark use, default is 0

  infer

  <model filename>        kmodel filename
  <output path>           output path
  --dataset <dataset path>
                          dataset path
  --dataset-format <dataset format>
                          dataset format, e.g. image|raw, default is image
  --input-layout <input layout>
                          input layout, e.g NCHW|NHWC, default is NCHW
```

### 描述

`ncc` 是 nncase 的命令行工具。它有两个命令： `compile` 和 `infer`。

`compile` 命令将你训练好的模型 (`.tflite`, `.caffemodel`, `.onnx`) 编译到 `.kmodel`。



- `-i, --input-format` 用来指定输入模型的格式。nncase 现在支持 `tflite`、`caffe` 和 `onnx` 输入格式。
- `-t, --target` 用来指定你想要你的模型在哪种目标设备上运行。`cpu` 几乎所有平台都支持的通用目标。`k210` 是 Kendryte K210 SoC 平台。如果你指定了 `k210`，这个模型就只能在 K210 运行或在你的 PC 上模拟运行。
- `<input file>` 用于指定输入模型文件
- `--input-prototxt`用于指定caffe模型的prototxt文件
- `<output file>` 用于指定输出模型文件
- `--output-arrays `用于指定输出结点的名称
- `--quant-type` 用于指定数据的量化类型, 如`uint8`/`int8`, 默认是`uint8`
- `--w-quant-type` 用于指定权重的量化类型, 如`uint8`/`int8`, 默认是`uint8`
- `--use-mse-quant-w`指定是否使用最小化mse(mean-square error, 均方误差)算法来量化权重.
- `--dataset` 用于提供量化校准集来量化你的模型。你需要从训练集中选择几百到上千个数据放到这个目录里。
- `--dataset-format` 用于指定量化校准集的格式。默认是 `image`，nncase 将使用 `opencv` 读取你的图片，并自动缩放到你的模型输入需要的尺寸。如果你的输入有 3 个通道，ncc 会将你的图片转换为值域是 [0,1] 布局是 `NCHW` 的张量。如果你的输入只有 1 个通道，ncc 会灰度化你的图片。如果你的数据集不是图片（例如音频或者矩阵），把它设置为 `raw`。这种场景下你需要把你的数据集转换为 float 张量的二进制文件。
- `--calibrate-method` 用于设置量化校准方法，它被用来选择最优的激活函数值域。默认值是 `no_clip`，ncc 会使用整个激活函数值域。如果你需要更好的量化结果，你可以使用 `l2`，但它需要花更长的时间寻找最优值域。
- `--preprocess`指定是否预处理, 添加后表示开启预处理
- `--swapRB`指定**预处理时**是否交换红和蓝两个通道数据, 用于实现RGB2BGR或BGR2RGB功能
- `--mean`指定**预处理时**标准化参数均值,例如添加`--mean "0.1 2.3 33.1f"`用于设置三个通道的均值.
- `--std`指定**预处理时**标准化参数方差,例如添加`--std "1. 2. 3."`用于设置三个通道的方差.
- `--input-range`指定输入数据反量化后的数据范围,例如添加`--input-range "0.1 2."`设置反量化的范围为`[0.1~2]`.
- `--input-shape`指定输入数据的形状. 若与模型的输入形状不同, 则预处理时会做resize/pad等处理, 例如添加`--input-shape "1 1 28 28"`指明当前输入图像尺寸.
- `--letterbox-value`用于指定预处理时pad填充的值.
- `--input-type` 用于指定推理时输入的数据类型。如果 `--input-type` 是 `uint8`，推理时你需要提供 RGB888 uint8 张量。如果 `--input-type` 是 `float`，你则需要提供 RGB float 张量.
- `--output-type` 用于指定推理时输出的数据类型。如`float`/`uint8`,  `uint8`仅在量化模型时才有效. 默认是`float`
- `--input-layout`用于指定输入数据的layout. 若输入数据的layout与模型的layout不同, 预处理会添加transpose进行转换.
- `--output-layout`用于指定输出数据的layout
- `--is-fpga`指定编译后的kmodel是否运行在fpga上
- `--dump-ir` 是一个调试选项。当它打开时 ncc 会在工作目录产生一些 `.dot` 文件。你可以使用 `Graphviz` 或 [Graphviz Online](https://dreampuf.github.io/GraphvizOnline) 来查看这些文件。
- `--dump-asm` 是一个调试选项。当它打开时 ncc 会生成硬件指令文件compile.text.asm
- `--dump-quant-error`是一个调试选项, 用于dump量化错误信息
- `--dump-dir`是一个调试选项, 用于指定dump目录.
- `--benchmark-only`是一个调试选项, 用于指定编译后的kmodel用于benchmark.



`infer` 命令可以运行你的 kmodel，通常它被用来调试。ncc 会将你模型的输出张量按 `NCHW` 布局保存到 `.bin` 文件。

- `<input file>` kmodel 的路径。
- `<output path>` ncc 输出目录。
- `--dataset` 测试集路径。
- `--dataset-format`和`--input-layout`同 `compile` 命令中的含义。
