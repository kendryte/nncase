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