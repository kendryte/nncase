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
- `--dataset-format` is to set the format of the calibration dataset. Default is `image`, nncase will use `opencv` to read your images and autoscale to the desired input size of your model. If the input has 3 channels, ncc will convert images to RGB float format [0,1]. If the input has only 1 channel, ncc will grayscale your images. Set to `raw` if your dataset is not image dataset for example, audio or matrices. In this scenario you should convert your dataset to raw binaries which contains float arrays. You only need this option when you set `--inference-type` to `uint8`.
- `--input-std` and `--input-mean` is to set the preprocess method on your calibration dataset. As said above, ncc will firstly convert images to RGB float format [0,1], then ncc will normalize your images using `y = (x - mean) / std` formula. There is a table for reference.

| Input range | --input-std | --input-mean |
|-------|------------------ |------------- |
| [0,1] (default) | 1 | 0 |
| [-1,1] | 0.5 | 0.5 |
| [0,255] | 0.0039216 | 0 |
- `--input-type` is to set your desired input data type when do inference. Default is equal to inference type. If `--input-type` is `uint8`, for example you should provide RGB888 uint8 arrays when you do inference. If `--input-type` is `float`, you shold provide RGB float arrays instead.
- `--max-allocator-solve-secs` is to limit the maximum solving time when do the best fit allocation search. If the search time exceeded, ncc will fallback to use the first fit method. Default is 60 secs, set to 0 to disable search.
- `--dump-ir` is a debug option. When it's on, ncc will produce some `.dot` graph files to the working directory. You can use `Graphviz` or [Graphviz Online](https://dreampuf.github.io/GraphvizOnline) to view these files.

`infer` command can run your kmodel, and it's often used as debug purpose.
- `<input file>` is your kmodel path.
- `<output path>` is the output directory ncc will produce to.

### 描述