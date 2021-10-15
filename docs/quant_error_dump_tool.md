## Intro

We have provided an embedded tool helping users to understand quantize error for each layer. Users could enable or disable this function for both ncc and python interface.

## Usage

### ncc

```
DESCRIPTION
specific "--dump-quant-error" argument for ncc compile and run ncc compile

EXAMPLE
ncc compile -i tflite -t k510 test.tflite test.cmodel --dataset ./calib_data --dump-ir --dump-asm
    --dump-dir ./dump_dir --dump-quant-error --input-layout NHWC --output-layout NHWC
```

### python

```
DESCRIPTION
add "compile_options.dump_quant_error = True" for "compile_options" in python script and run python script

COMPILE OPTIONS EXAMPLE
    compile_options = nncase.CompileOptions()
    compile_options.target = target
    compile_options.dump_ir = True
    compile_options.dump_asm = True
    compile_options.dump_dir = 'tmp'
    compile_options.dump_quant_error = True
```

When the tool is enabled, users could check quantize error information from terminal or a dump file under "dump_dir".

**sample code for compile with "dump_quant_error":**

```python
import nncase

def read_model_file(model_file):
    with open(model_file, 'rb') as f:
        model_content = f.read()
    return model_content

def main():
    model='examples/mobilenetv1/data/model_f32.tflite'
    target = 'k510'

    # compile_options
    compile_options = nncase.CompileOptions()
    compile_options.target = target
    compile_options.dump_ir = True
    compile_options.dump_asm = True
    compile_options.dump_dir = 'tmp'
    compile_options.dump_quant_error = True

    # compiler
    compiler = nncase.Compiler(compile_options)

    # import_options
    import_options = nncase.ImportOptions()

    # import
    model_content = read_model_file(model)
    compiler.import_tflite(model_content, import_options)

    # compile
    compiler.compile()

    # kmodel
    kmodel = compiler.gencode_tobytes()
    with open('test.kmodel', 'wb') as f:
        f.write(kmodel)

if __name__ == '__main__':
    main()

```

**quant error dump output:**
layer name: tfl.conv_2d/in_tp/store   cosine: 1
layer name: tfl.conv_2d/conv/conv2d   cosine: 1.00038
layer name: tfl.depthwise_conv_2d_0/conv2d   cosine: 0.998613
layer name: tfl.conv_2d1/conv/conv2d   cosine: 0.994524
layer name: tfl.depthwise_conv_2d1_0/conv2d   cosine: 1.00007
layer name: tfl.conv_2d2/conv/conv2d   cosine: 0.999273
layer name: tfl.depthwise_conv_2d2_0/conv2d   cosine: 0.999846
layer name: tfl.conv_2d3/conv/conv2d   cosine: 0.999506
layer name: tfl.depthwise_conv_2d3_0/conv2d   cosine: 0.999811
layer name: tfl.conv_2d4/conv/conv2d   cosine: 0.999844
layer name: tfl.depthwise_conv_2d4_0/conv2d   cosine: 0.999612
layer name: tfl.conv_2d5/conv/conv2d   cosine: 0.999411
layer name: tfl.depthwise_conv_2d5_0/conv2d   cosine: 0.999816
layer name: tfl.conv_2d6/conv/conv2d   cosine: 0.999474
layer name: tfl.depthwise_conv_2d6_0/conv2d   cosine: 0.999033
layer name: tfl.conv_2d7/conv/conv2d   cosine: 0.999206
layer name: tfl.depthwise_conv_2d7_0/conv2d   cosine: 0.999221
layer name: tfl.conv_2d8/conv/conv2d   cosine: 0.999041
layer name: tfl.depthwise_conv_2d8_0/conv2d   cosine: 0.999057
layer name: tfl.conv_2d9/conv/conv2d   cosine: 0.99843
layer name: tfl.depthwise_conv_2d9_0/conv2d   cosine: 0.998911
layer name: tfl.conv_2d10/conv/conv2d   cosine: 0.997248
layer name: tfl.depthwise_conv_2d10_0/conv2d   cosine: 0.998415
layer name: tfl.conv_2d11/conv/conv2d   cosine: 0.996701
layer name: tfl.depthwise_conv_2d11_0/conv2d   cosine: 0.999639
layer name: tfl.conv_2d12/conv/conv2d   cosine: 0.997942
layer name: tfl.depthwise_conv_2d12_0/conv2d   cosine: 0.999106
layer name: StatefulPartitionedCall:0/conv/conv2d   cosine: 0.972087
