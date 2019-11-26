## Usage
## 使用方法

### Comannd line
### 命令行
```
DESCRIPTION
NNCASE model compiler and inference tool.

SYNOPSIS
    ncc compile <input file> <output file> -i <input format> [-o <output
        format>] [-t <target>] [--dataset <dataset path>] [--dataset-format
        <dataset format>] [--inference-type <inference type>] [--input-mean
        <input mean>] [--input-std <input std>] [--dump-ir] [--input-type <input
        type>] [--max-allocator-solve-secs <max allocator solve secs>] [-v]

    ncc infer <input file> <output path> --dataset <dataset path>
        [--dataset-format <dataset format>] [--input-mean <input mean>]
        [--input-std <input std>] [-v]

OPTIONS
    compile
        <input file>        input file
        <output file>       output file
        -i, --input-format  input file format: e.g. tflite, caffe
        -o, --output-format output file format: e.g. kmodel, default is kmodel
        -t, --target        target arch: e.g. cpu, k210, default is k210
        --dataset           calibration dataset, used in post quantization
        --dataset-format    datset format: e.g. image, raw default is image
        --inference-type    inference type: e.g. float, uint8 default is uint8
        --input-mean        input mean, default is 0.000000
        --input-std         input std, default is 1.000000
        --dump-ir           dump nncase ir to .dot files
        --input-type        input type: e.g. default, float, uint8, default
                            means equal to inference type

        --max-allocator-solve-secs
                            max optimal layout solve time in secs used by
                            allocators, 0 means don't use solver, default is 60

    infer
        <input file>        input kmodel
        <output path>       inference result output directory
        --dataset           input dataset to inference
        --dataset-format    datset format: e.g. image, raw default is image
        --input-mean        input mean, default is 0.000000
        --input-std         input std, default is 1.000000

    -v, --version           show version
```

### Description

`ncc` is the nncase command line tool. It has two commands: `compile` and `infer`.

`compile` command compile your trained models (`.tflite`, `.caffemodel`) to `.kmodel`.
- `<input file>` is your input model path.
- `<output file>` is the output model path.
- `-i, --input-format` option is used to specify the input model format. nncase supports `tflite` and `caffe` input model currently.
- `-o, --output-format` option is used to sepecify the output model format. You have only one choice: `kmodel` currently.
- `-t, --target` option is used to set your desired target device to run the model. `cpu` is the most general target that almost every platform should support. `k210` is the Kendryte K210 SoC platform. If you set this option to `k210`, this model can only run on K210 or be emulated on your PC.
- `--inference-type` Set to `float` if you want precision, but you need more memory and lost K210 KPU acceleration. Set to `uint8` if you want KPU acceleration and fast speed and you need to provide a quantization calibration dataset to quantize your models later.
- `--dataset` is to provide your quantization calibration dataset to quantize your models. You should put hundreds or thousands of data in training set to this directory. You only need this option when you set `--inference-type` to `uint8`.
- `--dataset-format` is to set the format of the calibration dataset. Default is `image`, nncase will use `opencv` to read your images and autoscale to the desired input size of your model. If the input has 3 channels, ncc will convert images to RGB float tensors [0,1] in `NCHW` layout. If the input has only 1 channel, ncc will grayscale your images. Set to `raw` if your dataset is not image dataset for example, audio or matrices. In this scenario you should convert your dataset to raw binaries which contains float tensors. You only need this option when you set `--inference-type` to `uint8`.
- `--input-std` and `--input-mean` is to set the preprocess method on your calibration dataset. As said above, ncc will firstly convert images to RGB float tensors [0,1] in `NCHW` layout, then ncc will normalize your images using `y = (x - mean) / std` formula. There is an arguments table for reference.

| Input range | --input-std | --input-mean |
|-------|------------------ |------------- |
| [0,1] (default) | 1 | 0 |
| [-1,1] | 0.5 | 0.5 |
| [0,255] | 0.0039216 | 0 |
- `--input-type` is to set your desired input data type when do inference. Default is equal to inference type. If `--input-type` is `uint8`, for example you should provide RGB888 uint8 tensors when you do inference. If `--input-type` is `float`, you shold provide RGB float tensors instead.
- `--max-allocator-solve-secs` is to limit the maximum solving time when do the best fit allocation search. If the search time exceeded, ncc will fallback to use the first fit method. Default is 60 secs, set to 0 to disable search.
- `--dump-ir` is a debug option. When it's on, ncc will produce some `.dot` graph files to the working directory. You can use `Graphviz` or [Graphviz Online](https://dreampuf.github.io/GraphvizOnline) to view these files.

`infer` command can run your kmodel, and it's often used as debug purpose. ncc will save the model's output tensors to `.bin` files in `NCHW` layout.
- `<input file>` is your kmodel path.
- `<output path>` is the output directory ncc will produce to.
- `--dataset` is the test set directory.
- `--dataset-format`, `--input-std` and `--input-mean` has the same meaning as in `compile` command.

### 描述

`ncc` 是 nncase 的命令行工具。它有两个命令： `compile` 和 `infer`。

`compile` 命令将你训练好的模型 (`.tflite`, `.caffemodel`) 编译到 `.kmodel`。
- `<input file>` 是你输入模型的路径。
- `<output file>` 是输出模型的路径。
- `-i, --input-format` 用来指定输入模型的格式。nncase 现在支持 `tflite` 和 `caffe` 输入格式。
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
- `--input-type` 用于设置推理时输入的数据类型。默认和 inference type 相同。如果 `--input-type` 是 `uint8`，推理时你需要提供 RGB888 uint8 张量。如果 `--input-type` 是 `float`，你则需要提供 RGB float 张量。
- `--max-allocator-solve-secs` 用于限制 ncc 做最优分配时的最大搜索时间。如果搜索超过了这个时间，ncc 会退而使用 first fit 算法。默认是 60 秒，如果要禁用搜索请设置为 0。
- `--dump-ir` 是一个调试选项。当它打开时 ncc 会在工作目录产生一些 `.dot` 文件。你可以使用 `Graphviz` 或 [Graphviz Online](https://dreampuf.github.io/GraphvizOnline) 来查看这些文件。

`infer` 命令可以运行你的 kmodel，通常它被用来调试。ncc 会将你模型的输出张量按 `NCHW` 布局保存到 `.bin` 文件。
- `<input file>` kmodel 的路径。
- `<output path>` ncc 输出目录。
- `--dataset` 测试集路径。
- `--dataset-format`, `--input-std` 和 `--input-mean` 同 `compile` 命令中的含义。