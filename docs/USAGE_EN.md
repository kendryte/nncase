[toc]

# Overview

nncase provides both python wheel package and ncc client to compile your neural models.

- nncase wheel package can be downloaded at [nncase release](https://github.com/kendryte/nncase/releases)
- For ncc client, you should git clone nncase repository and then build it by yourself.

# nncase python APIs

nncase provides Python APIs to compile neural network model and inference on your PC.

## Installation

The nncase toolchain compiler consists of nncase and plug-in wheel packages.

- Both nncase and plug-in wheel packages are released at [nncase github](https://github.com/kendryte/nncase/releases)
- Nncase wheel package supports Python 3.6/3.7/3.8/3.9/3.10, You can download it according to your operating system and Python version.
- The plug-in wheel package does not depend on Python version, you can install it directly.

You can make use of [nncase docker image](https://github.com/kendryte/nncase/blob/master/docs/build.md)(Ubuntu 20.04 + Python 3.8) if you do not have Ubuntu development.

```shell
$ cd /path/to/nncase_sdk
$ docker pull registry.cn-hangzhou.aliyuncs.com/kendryte/nncase:latest
$ docker run -it --rm -v `pwd`:/mnt -w /mnt registry.cn-hangzhou.aliyuncs.com/kendryte/nncase:latest /bin/bash -c "/bin/bash"
```



### cpu/K210

- Download nncase wheel package and then install it.

```
root@2b11cc15c7f8:/mnt# wget -P x86_64 https://github.com/kendryte/nncase/releases/download/v1.8.0/nncase-1.8.0.20220929-cp38-cp38-manylinux_2_24_x86_64.whl

root@2b11cc15c7f8:/mnt# pip3 install x86_64/*.whl
```



### K510

- Download both nncase and nncase_k510 wheel packages and then install them.

```shell
root@2b11cc15c7f8:/mnt# wget -P x86_64 https://github.com/kendryte/nncase/releases/download/v1.8.0/nncase-1.8.0.20220929-cp38-cp38-manylinux_2_24_x86_64.whl

root@2b11cc15c7f8:/mnt# wget -P x86_64 https://github.com/kendryte/nncase/releases/download/v1.8.0/nncase_k510-1.8.0.20220930-py2.py3-none-manylinux_2_24_x86_64.whl

root@2b11cc15c7f8:/mnt# pip3 install x86_64/*.whl
```



### Check nncase version

```python
root@469e6a4a9e71:/mnt# python3
Python 3.8.10 (default, Jun  2 2021, 10:49:15)
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import _nncase
>>> print(_nncase.__version__)
1.8.0-55be52f
```



## nncase compile model APIs

### CompileOptions

#### Description

CompileOptions is used to configure compile options for nncase.

#### Definition

```python
py::class_<compile_options>(m, "CompileOptions")
    .def(py::init())
    .def_readwrite("target", &compile_options::target)
    .def_readwrite("quant_type", &compile_options::quant_type)
    .def_readwrite("w_quant_type", &compile_options::w_quant_type)
    .def_readwrite("use_mse_quant_w", &compile_options::use_mse_quant_w)
    .def_readwrite("split_w_to_act", &compile_options::split_w_to_act)
    .def_readwrite("preprocess", &compile_options::preprocess)
    .def_readwrite("swapRB", &compile_options::swapRB)
    .def_readwrite("mean", &compile_options::mean)
    .def_readwrite("std", &compile_options::std)
    .def_readwrite("input_range", &compile_options::input_range)
    .def_readwrite("output_range", &compile_options::output_range)
    .def_readwrite("input_shape", &compile_options::input_shape)
    .def_readwrite("letterbox_value", &compile_options::letterbox_value)
    .def_readwrite("input_type", &compile_options::input_type)
    .def_readwrite("output_type", &compile_options::output_type)
    .def_readwrite("input_layout", &compile_options::input_layout)
    .def_readwrite("output_layout", &compile_options::output_layout)
    .def_readwrite("model_layout", &compile_options::model_layout)
    .def_readwrite("is_fpga", &compile_options::is_fpga)
    .def_readwrite("dump_ir", &compile_options::dump_ir)
    .def_readwrite("dump_asm", &compile_options::dump_asm)
    .def_readwrite("dump_quant_error", &compile_options::dump_quant_error)
    .def_readwrite("dump_dir", &compile_options::dump_dir)
    .def_readwrite("benchmark_only", &compile_options::benchmark_only);
```

The details of all attributes are following.

| Attribute        | Data Type | *Required* | Description                                                                                                                                                                                             |
| ---------------- | --------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| target           | string    | Y            | Specify the compile target,  such as 'k210', 'k510'                                                                                                                                                     |
| quant_type       | string    | N            | Specify the quantization type for input data , such as 'uint8', 'int8', 'int16'                                                                                                                         |
| w_quant_type     | string    | N            | Specify the quantization type for weight , such as 'uint8'(by default), 'int8', 'int16'                                                                                                                 |
| use_mse_quant_w  | bool      | N            | Specify whether use  mean-square error when quantizing weight                                                                                                                                           |
| split_w_to_act   | bool      | N            | Specify whether split weight into activation                                                                                                                                            |
| preprocess       | bool      | N            | Whether enable preprocess, False by default                                                                                                                                                             |
| swapRB           | bool      | N            | Whether swap red and blue channel for RGB data(from RGB to BGR or from BGR to RGB), False by default                                                                                                    |
| mean             | list      | N            | Normalize mean value for preprocess, [0, 0, 0] by default                                                                                                                                               |
| std              | list      | N            | Normalize std value for preprocess, [1, 1, 1] by default                                                                                                                                                |
| input_range      | list      | N            | The float range for dequantized input data, [0，1] by default                                                                                                                                           |
| output_range | list | N | The float range for quantized output data,  [ ] by default |
| input_shape      | list      | N            | Specify the shape of input data.  input_shape should be consistent with input _layout.  There will be letterbox  operations(Such as resize/pad) if input_shape is not the same as input shape of model. |
| letterbox_value  | float     | N            | Specify the pad value of letterbox during preprocess.                                                                                                                                                   |
| input_type       | string    | N            | Specify the data type of input data, 'float32' by default.                                                                                                                                              |
| output_type      | string    | N            | Specify the data type of output data, 'float32' by default.                                                                                                                                             |
| input_layout     | string    | N            | Specify the layout of input data, such as 'NCHW', 'NHWC'.  Nncase will insert transpose operation if input_layout is different with the layout of model.                                                |
| output_layout    | string    | N            | Specify the layout of output data, such as 'NCHW', 'NHWC'.  Nncase will insert transpose operation if output_layout is different with the layout of model.                                              |
| model_layout     | string    | N            | Specific the layout of model when the layout of tflite model is "NCHW" and the layout of Onnx model or Caffe model is "NHWC", default is empty.                                                         |
| is_fpga          | bool      | N            | Specify the generated kmodel is used for fpga or not, False by default.                                                                                                                                 |
| dump_ir          | bool      | N            | Specify whether dump IR, False by default.                                                                                                                                                              |
| dump_asm         | bool      | N            | Specify whether dump asm file, False by default.                                                                                                                                                        |
| dump_quant_error | bool      | N            | Specify whether dump quantization error, False by default.                                                                                                                                              |
| dump_dir         | string    | N            | Specify dump directory                                                                                                                                                                                  |
| benchmark_only   | bool      | N            | Specify whether the generated kmodel is used for benchmark, False by default.                                                                                                                           |

> 1. Both mean and std are floating numbers to normalize.
> 2. input_range is the range for floating numbers. If the input_type is uint8, input_range means the dequantized range of uint8.
> 3. input_shape should be consistent with onput_layout. Take [1，224，224，3] for example.  If input_layout is 'NCHW'，input_shape should be [1,3,224,224], or input_shape should be [1,224,224,3];
>
> Examples
>
> 1. input_type is uint8，range is [0, 255]，input_range is also [0, 255]，so preprocess will convert input data from uint8 to float32.
> 2. input_type is uint8，range is [0, 255]，input_range is [0, 1]，so preprocess will dequantize the input data from uint8 to float32。

#### Example

```python
# compile_options
compile_options = nncase.CompileOptions()
compile_options.target = target
compile_options.input_type = 'float32'  # or 'uint8' 'int8'
compile_options.output_type = 'float32'  # or 'uint8' 'int8'. Only work in PTQ
compile_options.output_range = []  # Only work in PTQ and output type is not "float32"
compile_options.preprocess = True  # if False, the args below will unworked
compile_options.swapRB = True
compile_options.input_shape = [1, 224, 224, 3]  # keep layout same as input layout
compile_options.input_layout = 'NHWC'
compile_options.output_layout = 'NHWC'
compile_options.model_layout = ''  # default is empty. Specific it when tflite model with "NCHW" layout and Onnx(Caffe) model with "NHWC" layout
compile_options.mean = [0, 0, 0]
compile_options.std = [1, 1, 1]
compile_options.input_range = [0, 1]
compile_options.letterbox_value = 114.  # pad what you want
compile_options.dump_ir = True
compile_options.dump_asm = True
compile_options.dump_dir = 'tmp'
```

### ImportOptions

#### Description

ImportOptions is used to configure import options.

#### Definition

```python
py::class_<import_options>(m, "ImportOptions")
    .def(py::init())
    .def_readwrite("output_arrays", &import_options::output_arrays);
```

The details of all attributes are following.

| Attribute     | Data Type | *Required* | Description       |
| ------------- | --------- | ------------ | ----------------- |
| output_arrays | string    | N            | output array name |

#### Example

```python
# import_options
import_options = nncase.ImportOptions()
import_options.output_arrays = 'output' # Your output node name
```

### PTQTensorOptions

#### Description

PTQTensorOptions is used to configure PTQ options.

#### Definition

```python
py::class_<ptq_tensor_options>(m, "PTQTensorOptions")
    .def(py::init())
    .def_readwrite("calibrate_method", &ptq_tensor_options::calibrate_method)
    .def_readwrite("samples_count", &ptq_tensor_options::samples_count)
    .def("set_tensor_data", [](ptq_tensor_options &o, py::bytes bytes) {
        uint8_t *buffer;
        py::ssize_t length;
        if (PyBytes_AsStringAndSize(bytes.ptr(), reinterpret_cast<char **>(&buffer), &length))
            throw std::invalid_argument("Invalid bytes");
        o.tensor_data.assign(buffer, buffer + length);
    });
```

The details of all attributes are following.

| Attribute        | Data Type | Required | Description                                                                                                       |
| ---------------- | --------- | -------- | ----------------------------------------------------------------------------------------------------------------- |
| calibrate_method | string    | N        | Specify calibrate method, such as 'no_clip', 'l2', 'kld_m0', 'kld_m1', 'kld_m2' and 'cdf',  'no_clip' by default. |
| samples_count    | int       | N        | Specify the number of samples.                                                                                    |

#### set_tensor_data()

##### Description

Set data for tensor.

##### Definition

```python
set_tensor_data(calib_data)
```

##### Parameters

| Attribute  | Data Type | Required | Description               |
| ---------- | --------- | -------- | ------------------------- |
| calib_data | byte[]    | Y        | The data for calibrating. |

##### Returns

N/A

##### Example

```python
# ptq_options
ptq_options = nncase.PTQTensorOptions()
ptq_options.samples_count = cfg.generate_calibs.batch_size
ptq_options.set_tensor_data(np.asarray([sample['data'] for sample in self.calibs]).tobytes())
```

### Compiler

#### Description

Compiler is used to compile models.

#### Definition

```python
py::class_<compiler>(m, "Compiler")
    .def(py::init(&compiler::create))
    .def("import_tflite", &compiler::import_tflite)
    .def("import_onnx", &compiler::import_onnx)
    .def("import_caffe", &compiler::import_caffe)
    .def("compile", &compiler::compile)
    .def("use_ptq", py::overload_cast<ptq_tensor_options>(&compiler::use_ptq))
    .def("gencode", [](compiler &c, std::ostream &stream) { c.gencode(stream); })
    .def("gencode_tobytes", [](compiler &c) {
        std::stringstream ss;
        c.gencode(ss);
        return py::bytes(ss.str());
    })
    .def("create_evaluator", [](compiler &c, uint32_t stage) {
        auto &graph = c.graph(stage);
        return std::make_unique<graph_evaluator>(c.target(), graph);
    });
```

#### Example

```python
compiler = nncase.Compiler(compile_options)
```

#### import_tflite()

##### Description

Import tflite model.

##### Definition

```python
import_tflite(model_content, import_options)
```

##### Parameters

| Attribute      | Data Type     | Required | Description           |
| -------------- | ------------- | -------- | --------------------- |
| model_content  | byte[]        | Y        | The content of model. |
| import_options | ImportOptions | Y        | Import options        |

##### Returns

N/A

##### Example

```python
model_content = read_model_file(model)
compiler.import_tflite(model_content, import_options)
```

#### import_onnx()

##### Description

Import onnx model.

##### Definition

```python
import_onnx(model_content, import_options)
```

##### Parameters

| Attribute      | Data Type     | Required | Description           |
| -------------- | ------------- | -------- | --------------------- |
| model_content  | byte[]        | Y        | The content of model. |
| import_options | ImportOptions | Y        | Import options        |

##### Returns

N/A

##### Example

```python
model_content = read_model_file(model)
compiler.import_onnx(model_content, import_options)
```

#### import_caffe()

##### Description

Import caffe model.

> User should build and install caffe locally.

##### Definition

```python
import_caffe(caffemodel, prototxt)
```

##### Parameters

| Attribute  | Data Type | Required | Description                |
| ---------- | --------- | -------- | -------------------------- |
| caffemodel | byte[]    | Y        | The content of caffemodel. |
| prototxt   | byte[]    | Y        | The content of prototxt.   |

##### Returns

N/A

##### Example

```python
# import
caffemodel = read_model_file('test.caffemodel')
prototxt = read_model_file('test.prototxt')
compiler.import_caffe(caffemodel, prototxt)
```

#### use_ptq()

##### Description

Enable PTQ.

##### Definition

```python
use_ptq(ptq_options)
```

##### Parameters

| Attribute   | Data Type        | Required | Description  |
| ----------- | ---------------- | -------- | ------------ |
| ptq_options | PTQTensorOptions | Y        | PTQ options. |

##### Returns

N/A

##### Example

```python
compiler.use_ptq(ptq_options)
```

#### compile()

##### Description

Compile model.

##### Definition

```python
compile()
```

##### Parameters

N/A

##### Returns

N/A

##### Example

```python
compiler.compile()
```

#### gencode_tobytes()

##### Description

Generate byte code for model.

##### Definition

```python
gencode_tobytes()
```

##### Parameters

N/A

##### Returns

bytes[]

##### Example

```python
kmodel = compiler.gencode_tobytes()
with open(os.path.join(infer_dir, 'test.kmodel'), 'wb') as f:
    f.write(kmodel)
```

## Examples for compiling model

### Compile float32 model for tflite

```python
import nncase

def read_model_file(model_file):
    with open(model_file, 'rb') as f:
        model_content = f.read()
    return model_content

def main():
    model='examples/mobilenetv2/data/model_f32.tflite'
    target = 'k510'

    # compile_options
    compile_options = nncase.CompileOptions()
    compile_options.target = target
    compile_options.dump_ir = True
    compile_options.dump_asm = True
    compile_options.dump_dir = 'tmp'

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

### Compile float32 model for onnx

Use [ONNX Simplifier](https://github.com/daquexian/onnx-simplifier) to simplify onnx model before using nncase.

```python
import os
import onnxsim
import onnx
import nncase

def parse_model_input_output(model_file):
    onnx_model = onnx.load(model_file)
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    input_names = list(set(input_all) - set(input_initializer))
    input_tensors = [node for node in onnx_model.graph.input if node.name in input_names]

    # input
    inputs= []
    for _, e in enumerate(input_tensors):
        onnx_type = e.type.tensor_type
        input_dict = {}
        input_dict['name'] = e.name
        input_dict['dtype'] = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_type.elem_type]
        input_dict['shape'] = [(i.dim_value if i.dim_value != 0 else d) for i, d in zip(
            onnx_type.shape.dim, [1, 3, 224, 224])]
        inputs.append(input_dict)


    return onnx_model, inputs

def onnx_simplify(model_file):
    onnx_model, inputs = parse_model_input_output(model_file)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    input_shapes = {}
    for input in inputs:
        input_shapes[input['name']] = input['shape']

    onnx_model, check = onnxsim.simplify(onnx_model, input_shapes=input_shapes)
    assert check, "Simplified ONNX model could not be validated"

    model_file = os.path.join(os.path.dirname(model_file), 'simplified.onnx')
    onnx.save_model(onnx_model, model_file)
    return model_file


def read_model_file(model_file):
    with open(model_file, 'rb') as f:
        model_content = f.read()
    return model_content


def main():
    model_file = 'examples/mobilenetv2/data/mobilenetv2-7.onnx'
    target = 'k510'

    # onnx simplify
    model_file = onnx_simplify(model_file)

    # compile_options
    compile_options = nncase.CompileOptions()
    compile_options.target = target
    compile_options.dump_ir = True
    compile_options.dump_asm = True
    compile_options.dump_dir = 'tmp'

    # compiler
    compiler = nncase.Compiler(compile_options)

    # import_options
    import_options = nncase.ImportOptions()

    # import
    model_content = read_model_file(model_file)
    compiler.import_onnx(model_content, import_options)

    # compile
    compiler.compile()

    # kmodel
    kmodel = compiler.gencode_tobytes()
    with open('test.kmodel', 'wb') as f:
        f.write(kmodel)

if __name__ == '__main__':
    main()
```

### Compile float32 model for caffe

You can get caffe wheel package at [kendryte caffe](https://github.com/kendryte/caffe/releases).

```python
import nncase

def read_model_file(model_file):
    with open(model_file, 'rb') as f:
        model_content = f.read()
    return model_content

def main():
    target = 'k510'

    # compile_options
    compile_options = nncase.CompileOptions()
    compile_options.target = target
    compile_options.dump_ir = True
    compile_options.dump_asm = True
    compile_options.dump_dir = 'tmp'

    # compiler
    compiler = nncase.Compiler(compile_options)

    # import_options
    import_options = nncase.ImportOptions()

    # import
    caffemodel = read_model_file('examples/conv2d_caffe/test.caffemodel')
    prototxt = read_model_file('examples/conv2d_caffe/test.prototxt')
    compiler.import_caffe(caffemodel, prototxt)

    # compile
    compiler.compile()

    # kmodel
    kmodel = compiler.gencode_tobytes()
    with open('test.kmodel', 'wb') as f:
        f.write(kmodel)

if __name__ == '__main__':
    main()

```

### Compile float32 model for tflite with preprocessing

```python
import nncase

def read_model_file(model_file):
    with open(model_file, 'rb') as f:
        model_content = f.read()
    return model_content

def main():
    model='examples/mobilenetv2/data/model_f32.tflite'
    target = 'k510'

    # compile_options
    compile_options = nncase.CompileOptions()
    compile_options.target = target
    compile_options.input_type = 'float32'  # or 'uint8' 'int8'
    compile_options.preprocess = True # if False, the args below will unworked
    compile_options.swapRB = True
    compile_options.input_shape = [1,224,224,3] # keep layout same as input layout
    compile_options.input_layout = 'NHWC'
    compile_options.output_layout = 'NHWC'
    compile_options.mean = [0,0,0]
    compile_options.std = [1,1,1]
    compile_options.input_range = [0,1]
    compile_options.letterbox_value = 114. # pad what you want
    compile_options.dump_ir = True
    compile_options.dump_asm = True
    compile_options.dump_dir = 'tmp'

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

### Compile uint8 model for tflite

```python
import nncase
import numpy as np

def read_model_file(model_file):
    with open(model_file, 'rb') as f:
        model_content = f.read()
    return model_content

def generate_data(shape, batch):
    shape[0] *= batch
    data = np.random.rand(*shape).astype(np.float32)
    return data

def main():
    model='examples/mobilenetv2/data/model_f32.tflite'
    input_shape = [1,224,224,3]
    target = 'k510'

    # compile_options
    compile_options = nncase.CompileOptions()
    compile_options.target = target
    compile_options.input_type = 'float32'
    compile_options.input_layout = 'NHWC'
    compile_options.output_layout = 'NHWC'
    compile_options.dump_ir = True
    compile_options.dump_asm = True
    compile_options.dump_dir = 'tmp'

    # compiler
    compiler = nncase.Compiler(compile_options)

    # import_options
    import_options = nncase.ImportOptions()

    # quantize model
    compile_options.quant_type = 'uint8' # or 'int8' 'int16'

    # ptq_options
    ptq_options = nncase.PTQTensorOptions()
    ptq_options.samples_count = 10
    ptq_options.set_tensor_data(generate_data(input_shape, ptq_options.samples_count).tobytes())

    # import
    model_content = read_model_file(model)
    compiler.import_tflite(model_content, import_options)

    # compile
    compiler.use_ptq(ptq_options)
    compiler.compile()

    # kmodel
    kmodel = compiler.gencode_tobytes()
    with open('test.kmodel', 'wb') as f:
        f.write(kmodel)

if __name__ == '__main__':
    main()

```

## Deploy nncase runtime

### Inference on K210 development board

1. Download [SDK](https://github.com/kendryte/kendryte-standalone-sdk)

      ```shell
      $ git clone https://github.com/kendryte/kendryte-standalone-sdk.git
      $ cd kendryte-standalone-sdk
      $ export KENDRYTE_WORKSPACE=`pwd`
      ```

2. Download the cross-compile toolchain and extract it

   ```shell
   $ wget https://github.com/kendryte/kendryte-gnu-toolchain/releases/download/v8.2.0-20190409/kendryte-toolchain-ubuntu-amd64-8.2.0-20190409.tar.xz -O $KENDRYTE_WORKSPACE/kendryte-toolchain.tar.xz
   $ cd $KENDRYTE_WORKSPACE
   $ mkdir toolchain
   $ tar -xf kendryte-toolchain.tar.xz -C ./toolchain
   ```

3. Update nncase runtime

   Download `k210-runtime.zip` from [Release](https://github.com/kendryte/nncase/releases) and extract it into [kendryte-standalone-sdk](https://github.com/kendryte/kendryte-standalone-sdk) 's `lib/nncase/v1`.

4. Compile App

   ```shell
   # 1.copy your programe into `$KENDRYTE_WORKSPACE/src`
   # e.g. copy ($NNCASE_WORK_DIR/examples/facedetect_landmark/k210/facedetect_landmark_example) into PATH_TO_SDK/src.
   $ cp -r $NNCASE_WORK_DIR/examples/facedetect_landmark/k210/facedetect_landmark_example $KENDRYTE_WORKSPACE/src/
   
   # 2. compile
   $ cd $KENDRYTE_WORKSPACE
   $ mkdir build
   $ cmake .. -DPROJ=facedetect_landmark_example -DTOOLCHAIN=$KENDRYTE_WORKSPACE/toolchain/kendryte-toolchain/bin && make
   ```

   `facedetect_landmark_example` and`FaceDETECt_landmark_example.bin` will be generated.

5. Write the program to the K210 development board

   ```shell
   # 1. Check available USB ports
   $ ls /dev/ttyUSB*
   # /dev/ttyUSB0 /dev/ttyUSB1
   
   # 2. Write your App by kflash
   $ kflash -p /dev/ttyUSB0 -t facedetect_landmark_example.bin
   ```

## nncase inference APIs

Nncase provides inference APIs to inference kmodel. You can make use of it to check the result with runtime for deep learning frameworks.

### MemoryRange

#### Description

MemoryRange is used to describe the range to memory.

#### Definition

```python
py::class_<memory_range>(m, "MemoryRange")
    .def_readwrite("location", &memory_range::memory_location)
    .def_property(
        "dtype", [](const memory_range &range) { return to_dtype(range.datatype); },
        [](memory_range &range, py::object dtype) { range.datatype = from_dtype(py::dtype::from_args(dtype)); })
    .def_readwrite("start", &memory_range::start)
    .def_readwrite("size", &memory_range::size);
```

The details of all attributes are following.

| Attribute | Data Type        | Required | Description                                                                                                      |
| --------- | ---------------- | -------- | ---------------------------------------------------------------------------------------------------------------- |
| location  | int              | N        | Specify the location of memory. 0 means input, 1 means output, 2 means rdata, 3 means data, 4 means shared_data. |
| dtype     | python data type | N        | data type                                                                                                        |
| start     | int              | N        | The start of memory                                                                                              |
| size      | int              | N        | The size of memory                                                                                               |

#### Example

```python
mr = nncase.MemoryRange()
```

### RuntimeTensor

#### Description

RuntimeTensor is used to describe the runtime tensor.

#### Definition

```python
py::class_<runtime_tensor>(m, "RuntimeTensor")
    .def_static("from_numpy", [](py::array arr) {
        auto src_buffer = arr.request();
        auto datatype = from_dtype(arr.dtype());
        auto tensor = host_runtime_tensor::create(
            datatype,
            to_rt_shape(src_buffer.shape),
            to_rt_strides(src_buffer.itemsize, src_buffer.strides),
            gsl::make_span(reinterpret_cast<gsl::byte *>(src_buffer.ptr), src_buffer.size * src_buffer.itemsize),
            [=](gsl::byte *) { arr.dec_ref(); })
                          .unwrap_or_throw();
        arr.inc_ref();
        return tensor;
    })
    .def("copy_to", [](runtime_tensor &from, runtime_tensor &to) {
        from.copy_to(to).unwrap_or_throw();
    })
    .def("to_numpy", [](runtime_tensor &tensor) {
        auto host = tensor.as_host().unwrap_or_throw();
        auto src_map = std::move(hrt::map(host, hrt::map_read).unwrap_or_throw());
        auto src_buffer = src_map.buffer();
        return py::array(
            to_dtype(tensor.datatype()),
            tensor.shape(),
            to_py_strides(runtime::get_bytes(tensor.datatype()), tensor.strides()),
            src_buffer.data());
    })
    .def_property_readonly("dtype", [](runtime_tensor &tensor) {
        return to_dtype(tensor.datatype());
    })
    .def_property_readonly("shape", [](runtime_tensor &tensor) {
        return to_py_shape(tensor.shape());
    });
```

The details of all attributes are following.

| Attribute | Data Type | Required | Description             |
| --------- | --------- | -------- | ----------------------- |
| dtype     | int       | N        | The data type of tensor |
| shape     | list      | N        | The shape of tensor     |

#### from_numpy()

##### Description

Construct RuntimeTensor from numpy.ndarray

##### Definition

```python
from_numpy(py::array arr)
```

##### Parameters

| Attribute | Data Type     | Required | Description   |
| --------- | ------------- | -------- | ------------- |
| arr       | numpy.ndarray | Y        | numpy.ndarray |

##### Returns

RuntimeTensor

##### Example

```python
tensor = nncase.RuntimeTensor.from_numpy(self.inputs[i]['data'])
```

#### copy_to()

##### Description

Copy RuntimeTensor

##### Definition

```python
copy_to(RuntimeTensor to)
```

##### Parameters

| Attribute | Data Type     | Required | Description   |
| --------- | ------------- | -------- | ------------- |
| to        | RuntimeTensor | Y        | RuntimeTensor |

##### Returns

N/A

##### Example

```python
sim.get_output_tensor(i).copy_to(to)
```

#### to_numpy()

##### Description

Convert RuntimeTensor to numpy.ndarray.

##### Definition

```python
to_numpy()
```

##### Parameters

N/A

##### Returns

numpy.ndarray

##### Example

```python
arr = sim.get_output_tensor(i).to_numpy()
```

### Simulator

#### Description

Simulator is used to inference kmodel on PC.

#### Definition

```python
py::class_<interpreter>(m, "Simulator")
    .def(py::init())
    .def("load_model", [](interpreter &interp, gsl::span<const gsl::byte> buffer) { interp.load_model(buffer).unwrap_or_throw(); })
    .def_property_readonly("inputs_size", &interpreter::inputs_size)
    .def_property_readonly("outputs_size", &interpreter::outputs_size)
    .def("get_input_desc", &interpreter::input_desc)
    .def("get_output_desc", &interpreter::output_desc)
    .def("get_input_tensor", [](interpreter &interp, size_t index) { return interp.input_tensor(index).unwrap_or_throw(); })
    .def("set_input_tensor", [](interpreter &interp, size_t index, runtime_tensor tensor) { return interp.input_tensor(index, tensor).unwrap_or_throw(); })
    .def("get_output_tensor", [](interpreter &interp, size_t index) { return interp.output_tensor(index).unwrap_or_throw(); })
    .def("set_output_tensor", [](interpreter &interp, size_t index, runtime_tensor tensor) { return interp.output_tensor(index, tensor).unwrap_or_throw(); })
    .def("run", [](interpreter &interp) { interp.run().unwrap_or_throw(); });
```

The details of all attributes are following.

| Attribute    | Data Type | Required | Description            |
| ------------ | --------- | -------- | ---------------------- |
| inputs_size  | int       | N        | The number of inputs.  |
| outputs_size | int       | N        | The number of outputs. |

#### Example

```python
sim = nncase.Simulator()
```

#### load_model()

##### Description

Load kmodel.

##### Definition

```python
load_model(model_content)
```

##### Parameters

| Attribute     | Data Type | Required | Description        |
| ------------- | --------- | -------- | ------------------ |
| model_content | byte[]    | Y        | kmodel byte stream |

##### Returns

N/A

##### Example

```python
sim.load_model(kmodel)
```

#### get_input_desc()

##### Description

Get description for input.

##### Definition

```python
get_input_desc(index)
```

##### Parameters

| Attribute | Data Type | Required | Description          |
| --------- | --------- | -------- | -------------------- |
| index     | int       | Y        | The index for input. |

##### Returns

MemoryRange

##### Example

```python
input_desc_0 = sim.get_input_desc(0)
```

#### get_output_desc()

##### Description

Get description for output.

##### Definition

```python
get_output_desc(index)
```

##### Parameters

| Attribute | Data Type | Required | Description           |
| --------- | --------- | -------- | --------------------- |
| index     | int       | Y        | The index for output. |

##### Returns

MemoryRange

##### Example

```python
output_desc_0 = sim.get_output_desc(0)
```

#### get_input_tensor()

##### Description

Get the input runtime tensor with specified index.

##### Definition

```python
get_input_tensor(index)
```

##### Parameters

| Attribute | Data Type | Required | Description                 |
| --------- | --------- | -------- | --------------------------- |
| index     | int       | Y        | The index for input tensor. |

##### Returns

RuntimeTensor

##### Example

```python
input_tensor_0 = sim.get_input_tensor(0)
```

#### set_input_tensor()

##### Description

Set the input runtime tensor with specified index.

##### Definition

```python
set_input_tensor(index, tensor)
```

##### Parameters

| Attribute | Data Type     | Required | Description                 |
| --------- | ------------- | -------- | --------------------------- |
| index     | int           | Y        | The index for input tensor. |
| tensor    | RuntimeTensor | Y        | RuntimeTensor               |

##### Returns

N/A

##### Example

```python
sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(self.inputs[0]['data']))
```

#### get_output_tensor()

##### Description

Get the output runtime tensor with specified index.

##### Definition

```python
get_output_tensor(index)
```

##### Parameters

| Attribute | Data Type | Required | Description                  |
| --------- | --------- | -------- | ---------------------------- |
| index     | int       | Y        | The index for output tensor. |

##### Returns

RuntimeTensor

##### Example

```python
output_arr_0 = sim.get_output_tensor(0).to_numpy()
```

#### set_output_tensor()

##### Description

Set the RuntimeTensor with specified index.

##### Definition

```python
set_output_tensor(index, tensor)
```

##### Parameters

| Attribute | Data Type     | Required | Description                  |
| --------- | ------------- | -------- | ---------------------------- |
| index     | int           | Y        | The index for output tensor. |
| tensor    | RuntimeTensor | Y        | RuntimeTensor                |

##### Returns

N/A

##### Example

```python
sim.set_output_tensor(0, tensor)
```

#### run()

##### Description

Run kmodel for inferencing.

##### Definition

```python
run()
```

##### Parameters

N/A

##### Returns

N/A

##### Example

```python
sim.run()
```

# ncc

## Comannd line

```shell
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
        [--input-layout <input layout>] [--output-layout <output layout>] [--tcu-num <tcu number>]
        [--is-fpga] [--dump-ir] [--dump-asm] [--dump-quant-error] [--dump-import-op-range] [--dump-dir <dump directory>]
        [--dump-range-dataset <dataset path>] [--dump-range-dataset-format <dataset format>] [--benchmark-only]

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
                          post trainning quantize type, e.g uint8|int8|int16, default is uint8
  --w-quant-type <w quant type>
                          post trainning weights quantize type, e.g uint8|int8|int16, default is uint8
  --use-mse-quant-w       use min mse algorithm to refine weights quantilization or not, default is 0
  --dataset <dataset path>
                          calibration dataset, used in post quantization
  --dataset-format <dataset format>
                          datset format: e.g. image|raw, default is image
  --dump-range-dataset <dataset path>
                          dump import op range dataset
  --dump-range-dataset-format <dataset format>
                          datset format: e.g. image|raw, default is image
  --calibrate-method <calibrate method>
                          calibrate method: e.g. no_clip|l2|kld_m0|kld_m1|kld_m2|cdf, default is no_clip
  --preprocess            enable preprocess, default is 0
  --swapRB                swap red and blue channel, default is 0
  --mean <normalize mean> normalize mean, default is 0. 0. 0.
  --std <normalize std>   normalize std, default is 1. 1. 1.
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
  --tcu-num <tcu number>  tcu number, e.g 1|2|3|4, default is 0
  --is-fpga               use fpga parameters, default is 0
  --dump-ir               dump ir to .dot, default is 0
  --dump-asm              dump assembly, default is 0
  --dump-quant-error      dump quant error, default is 0
  --dump-import-op-range  dump import op range, default is 0
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

## Description

`ncc` is the nncase command line tool. It has two commands: `compile` and `infer`.

`compile` command compile your trained models (`.tflite`, `.caffemodel`, `.onnx`) to `.kmodel`.

- `-i, --input-format` option is used to specify the input model format. nncase supports `tflite`, `caffe` and `onnx` input model currently.
- `-t, --target` option is used to set your desired target device to run the model. `cpu` is the most general target that almost every platform should support. `k210` is the Kendryte K210 SoC platform. If you set this option to `k210`, this model can only run on K210 or be emulated on your PC.
- `<input file>` is your input model path.
- `--input-prototxt` is the prototxt file for caffe model.
- `<output file>` is the output model path.
- `--output-arrays` is the names of nodes to output.
- `--quant-type` is used to specify quantize type, such as `uint8` by default and `int8` and `int16`.
- `--w-quant-type` is used to specify quantize type for weight, such as `uint8` by default and `int8 `and `int16`.
- `--use-mse-quant-w ` is used to specify whether use minimize mse(mean-square error, mse) algorithm to quantize weight or not.
- `--dataset` is to provide your quantization calibration dataset to quantize your models. You should put hundreds or thousands of data in training set to this directory.
- `--dataset-format` is to set the format of the calibration dataset. Default is `image`, nncase will use `opencv` to read your images and autoscale to the desired input size of your model. If the input has 3 channels, ncc will convert images to RGB float tensors [0,1] in `NCHW` layout. If the input has only 1 channel, ncc will grayscale your images. Set to `raw` if your dataset is not image dataset for example, audio or matrices. In this scenario you should convert your dataset to raw binaries which contains float tensors.
- `--dump-range-dataset` is to provide your dump range dataset to dump each op data range of your models. You should put hundreds or thousands of data in training set to this directory.
- `--dump-range-dataset-format` is to set the format of the dump range dataset. Default is `image`, nncase will use `opencv` to read your images and autoscale to the desired input size of your model. If the input has 3 channels, ncc will convert images to RGB float tensors [0,1] in `NCHW` layout. If the input has only 1 channel, ncc will grayscale your images. Set to `raw` if your dataset is not image dataset for example, audio or matrices. In this scenario you should convert your dataset to raw binaries which contains float tensors.
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
- `--tcu-num` is used to configure the number of TCU. 0 means do not configure the number of TCU.
- `--is-fpga` is a debug option. It is used to specify whether the kmodel run on fpga or not.
- `--dump-ir` is a debug option. It is used to specify whether dump IR or not.
- `--dump-asm` is a debug option. It is used to specify whether dump asm file or not.
- `--dump-quant-error` is a debug option. It is used to specify whether dump quantization error information or not.
- `--dump-import-op-range` is a debug option. It is used to specify whether dump imported op data range or not, need to also specify dump-range-dataset if enabled.
- `--dump-dir` is used to specify dump directory.
- `--benchmark-only` is used to specify whether the kmodel is used for benchmark or not.

`infer` command can run your kmodel, and it's often used as debug purpose. ncc will save the model's output tensors to `.bin` files in `NCHW` layout.

- `<input file>` is your kmodel path.
- `<output path>` is the output directory ncc will produce to.
- `--dataset` is the test set directory.
- `--dataset-format` and `--input-layout` have the same meaning as in `compile` command.
