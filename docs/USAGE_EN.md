## Usage

### Comannd line
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
  --use-mse-quant-w       use mse to refine weights quantilization or not, default is 0
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

### Description

`ncc` is the nncase command line tool. It has two commands: `compile` and `infer`.

`compile` command compile your trained models (`.tflite`, `.caffemodel`, `.onnx`) to `.kmodel`.

- `-i, --input-format` option is used to specify the input model format. nncase supports `tflite`, `caffe` and `onnx` input model currently.
- `-t, --target` option is used to set your desired target device to run the model. `cpu` is the most general target that almost every platform should support. `k210` is the Kendryte K210 SoC platform. If you set this option to `k210`, this model can only run on K210 or be emulated on your PC.
- `<input file>` is your input model path.
- `--input-prototxt` is the prototxt file for caffe model.
- `<output file>` is the output model path.
- `--output-arrays` is the names of nodes to output.
- `--quant-type` is used to specify quantize type, such as `uint8` by default and `int8`.
- `--w-quant-type` is used to specify quantize type for weight, such as `uint8` by default and `int8`.
- `--use-mse-quant-w ` is used to specify whether use minimize mse(mean-square error, mse) algorithm to quantize weight or not.
- `--dataset` is to provide your quantization calibration dataset to quantize your models. You should put hundreds or thousands of data in training set to this directory.
- `--dataset-format` is to set the format of the calibration dataset. Default is `image`, nncase will use `opencv` to read your images and autoscale to the desired input size of your model. If the input has 3 channels, ncc will convert images to RGB float tensors [0,1] in `NCHW` layout. If the input has only 1 channel, ncc will grayscale your images. Set to `raw` if your dataset is not image dataset for example, audio or matrices. In this scenario you should convert your dataset to raw binaries which contains float tensors.
- `--calibrate-method` is to set your desired calibration method, which is used to select the optimal activation ranges. The default is `no_clip` in that ncc will use the full range of activations. If you want a better quantization result, you can use `l2` but it will take a longer time to find the optimal ranges.
- `--preprocess ` is used specify whether enable preprocessing or not.
- `--swapRB ` is used specify whether swap red and blue channel or not. You can use this flag to implement RGB2BGR or BGR2RGB feature.
- `--mean` is the mean values to be subtracted during preprocessing.
- `--std` is the std values to be divided during preprocessing.
- `--input-range` is the input range in float after dequantization.
- `--input-shape` is used to specify the shape of input data. If the input shape is different from the input shape of your model, the preprocess will add resize/pad ops automatically for the transformation.
- `--letterbox-value` is used to specify the pad values when pad is added during preprocessing.
- `--input-type` is to set your desired input data type when do inference. If `--input-type` is `uint8`, for example you should provide RGB888 uint8 tensors when you do inference. If `--input-type` is `float`, you should provide RGB float tensors instead.
- `--output-type` is the type of output data.
- `--input-layout` is the layout of input data.
- `--output-layout` is the layout of output data.
- `--is-fpga` is a debug option. It is used to specify whether the kmodel run on fpga or not.
- `--dump-ir` is a debug option. It is used to specify whether dump IR or not.
- `--dump-asm` is a debug option. It is used to specify whether dump asm file or not.
- `--dump-quant-error` is a debug option. It is used to specify whether dump quantization error information or not.
- `--dump-dir` is used to specify dump directory.
- `--benchmark-only` is used to specify whether the kmodel is used for benchmark or not.



`infer` command can run your kmodel, and it's often used as debug purpose. ncc will save the model's output tensors to `.bin` files in `NCHW` layout.

- `<input file>` is your kmodel path.
- `<output path>` is the output directory ncc will produce to.
- `--dataset` is the test set directory.
- `--dataset-format` and `--input-layout` have the same meaning as in `compile` command.
