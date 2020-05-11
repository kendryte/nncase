## 使用方法

### 命令行
```
DESCRIPTION
NNCASE model compiler and inference tool.

SYNOPSIS
    ncc compile <input file> <output file> -i <input format> [-o <output
        format>] [-t <target>] [--dataset <dataset path>] [--dataset-format
        <dataset format>] [--inference-type <inference type>] [--input-mean
        <input mean>] [--input-std <input std>] [--dump-ir] [--input-type <input
        type>] [--max-allocator-solve-secs <max allocator solve secs>]
        [--calibrate-method <calibrate method>] [-v]

    ncc infer <input file> <output path> --dataset <dataset path>
        [--dataset-format <dataset format>] [--input-mean <input mean>]
        [--input-std <input std>] [-v]

OPTIONS
    ncc compile <input file> <output file> -i <input format> [-o <output
        format>] [-t <target>] [--dataset <dataset path>] [--dataset-format
        <dataset format>] [--inference-type <inference type>] [--input-mean
        <input mean>] [--input-std <input std>] [--dump-ir]
        [--dump-weights-range] [--input-type <input type>]
        [--max-allocator-solve-secs <max allocator solve secs>]
        [--calibrate-method <calibrate method>] [--weights-quantize-threshold
        <weights quantize threshold>] [--no-quantized-binary] [-v]

    ncc infer <input file> <output path> --dataset <dataset path>
        [--dataset-format <dataset format>] [--input-mean <input mean>]
        [--input-std <input std>] [-v]

    ncc -h [-v]

OPTIONS
    compile
        <input file>        input file
        <output file>       output file
        -i, --input-format  input file format: e.g. tflite, caffe, onnx
        -o, --output-format output file format: e.g. kmodel, default is kmodel
        -t, --target        target arch: e.g. cpu, k210, default is k210
        --dataset           calibration dataset, used in post quantization
        --dataset-format    datset format: e.g. image, raw default is image
        --inference-type    inference type: e.g. float, uint8 default is uint8
        --input-mean        input mean, default is 0.000000
        --input-std         input std, default is 1.000000
        --dump-ir           dump nncase ir to .dot files
        --dump-weights-range
                            dump weights range

        --input-type        input type: e.g. default, float, uint8, default
                            means equal to inference type

        --max-allocator-solve-secs
                            max optimal layout solve time in secs used by
                            allocators, 0 means don't use solver, default is 60

        --calibrate-method  calibrate method: e.g. no_clip, l2, default is
                            no_clip

        --weights-quantize-threshold
                            the threshold to control quantizing op or not
                            according to it's weigths range, default is
                            32.000000

        --output-quantize-threshold
                            the threshold to control quantizing op or not
                            according to it's output size, default is 1024

        --no-quantized-binary
                            don't quantize binary ops

    infer
        <input file>        input kmodel
        <output path>       inference result output directory
        --dataset           input dataset to inference
        --dataset-format    datset format: e.g. image, raw default is image
        --input-mean        input mean, default is 0.000000
        --input-std         input std, default is 1.000000

    -h, --help              show help
    -v, --version           show version
```

### 描述

`ncc` 是 nncase 的命令行工具。它有两个命令： `compile` 和 `infer`。

`compile` 命令将你训练好的模型 (`.tflite`, `.caffemodel`, `.onnx`) 编译到 `.kmodel`。
- `<input file>` 是你输入模型的路径。
- `<output file>` 是输出模型的路径。
- `-i, --input-format` 用来指定输入模型的格式。nncase 现在支持 `tflite`、`caffe` 和 `onnx` 输入格式。
- `-o, --output-format` 用来指定输出模型的格式。你现在只有一个选项：`kmodel`。
- `-t, --target` 用来指定你想要你的模型在哪种目标设备上运行。`cpu` 几乎所有平台都支持的通用目标。`k210` 是 Kendryte K210 SoC 平台。如果你指定了 `k210`，这个模型就只能在 K210 运行或在你的 PC 上模拟运行。
- `--inference-type` 如果你需要精度，设置为 `float`，但你需要更多内存并且失去了 K210 KPU 的加速能力。如果你需要 KPU 加速和更快的执行速度，设置为 `uint8`，之后你需要提供量化校准集来量化你的模型。
- `--dataset` 用于提供量化校准集来量化你的模型。你需要从训练集中选择几百到上千个数据放到这个目录里。你只需要在设置 `--inference-type` 为 `uint8` 时提供这个参数。
- `--dataset-format` 用于指定量化校准集的格式。默认是 `image`，nncase 将使用 `opencv` 读取你的图片，并自动缩放到你的模型输入需要的尺寸。如果你的输入有 3 个通道，ncc 会将你的图片转换为值域是 [0,1] 布局是 `NCHW` 的张量。如果你的输入只有 1 个通道，ncc 会灰度化你的图片。如果你的数据集不是图片（例如音频或者矩阵），把它设置为 `raw`。这种场景下你需要把你的数据集转换为 float 张量的二进制文件。你只需要在设置 `--inference-type` 为 `uint8` 时提供这个参数。
- `--input-std` 和 `--input-mean` 用于指定量化校准集的预处理方法。如上所述 ncc 会将你的图片转换为值域是 [0,1] 布局是 `NCHW` 的张量，之后 ncc 会使用 `y = (x - mean) / std` 公式对数据进行归一化。这里有一张参数的参考表。

| 输入值域 | --input-std | --input-mean |
|-------|------------------ |------------- |
| [0,1] (默认) | 1 | 0 |
| [-1,1] | 0.5 | 0.5 |
| [0,255] | 0.0039216 | 0 |
- `--calibrate-method` 用于设置量化校准方法，它被用来选择最优的激活函数值域。默认值是 `no_clip`，ncc 会使用整个激活函数值域。如果你需要更好的量化结果，你可以使用 `l2`，但它需要花更长的时间寻找最优值域。
- `--input-type` 用于设置推理时输入的数据类型。默认和 inference type 相同。如果 `--input-type` 是 `uint8`，推理时你需要提供 RGB888 uint8 张量。如果 `--input-type` 是 `float`，你则需要提供 RGB float 张量。
- `--max-allocator-solve-secs` 用于限制 ncc 做最优分配时的最大搜索时间。如果搜索超过了这个时间，ncc 会退而使用 first fit 算法。默认是 60 秒，如果要禁用搜索请设置为 0。
- `--weights-quantize-threshold` 控制是否量化 conv2d 和 matmul weights 的阈值。如果 weights 的范围大于这个阈值，nncase 将不会量化它。
- `--output-quantize-threshold` 控制是否量化 conv2d 和 matmul weights 的阈值。如果输出的元素个数小于这个阈值，nncase 将不会量化它。
- `--no-quantized-binary` 禁用 quantized binary 算子，nncase 将总是使用 float binary 算子。
- `--dump-ir` 是一个调试选项。当它打开时 ncc 会在工作目录产生一些 `.dot` 文件。你可以使用 `Graphviz` 或 [Graphviz Online](https://dreampuf.github.io/GraphvizOnline) 来查看这些文件。
- `--dump-weights-range` 是一个调试选项。当它打开时 ncc 会打印出 conv2d weights 的范围。

`infer` 命令可以运行你的 kmodel，通常它被用来调试。ncc 会将你模型的输出张量按 `NCHW` 布局保存到 `.bin` 文件。
- `<input file>` kmodel 的路径。
- `<output path>` ncc 输出目录。
- `--dataset` 测试集路径。
- `--dataset-format`, `--input-std` 和 `--input-mean` 同 `compile` 命令中的含义。