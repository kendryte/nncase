# 概述

nncase目前提供了python wheel包和ncc客户端两种方法编译模型.

- nncase wheel包需要去[nncase release](https://github.com/kendryte/nncase/releases)获取
- ncc客户端需要用户下载并编译nncase

# nncase python APIs

nncase提供了Python APIs, 用于在PC上编译/推理深度学习模型.

## 安装

nncase工具链compiler部分包括nncase和插件包

- nncase 和插件包均在[nncase github](https://github.com/kendryte/nncase/releases)发布
- nncase wheel包支持Python 3.6/3.7/3.8/3.9/3.10, 用户可根据操作系统和Python选择相应版本下载 .
- 插件包不依赖Python版本, 可直接安装

用户若没有Ubuntu环境, 可使用[nncase docker](https://github.com/kendryte/nncase/blob/master/docs/build.md#docker)(Ubuntu 20.04 + Python 3.8)

```shell
$ cd /path/to/nncase_sdk
$ docker pull registry.cn-hangzhou.aliyuncs.com/kendryte/nncase:latest
$ docker run -it --rm -v `pwd`:/mnt -w /mnt registry.cn-hangzhou.aliyuncs.com/kendryte/nncase:latest /bin/bash -c "/bin/bash"
```



### cpu/K210

- 下载nncase wheel包, 直接安装即可.

```
root@2b11cc15c7f8:/mnt# wget -P x86_64 https://github.com/kendryte/nncase/releases/download/v1.8.0/nncase-1.8.0.20220929-cp38-cp38-manylinux_2_24_x86_64.whl

root@2b11cc15c7f8:/mnt# pip3 install x86_64/*.whl
```



### K510

- 分别下载nncase和nncase_k510插件包，再一起安装

```shell
root@2b11cc15c7f8:/mnt# wget -P x86_64 https://github.com/kendryte/nncase/releases/download/v1.8.0/nncase-1.8.0.20220929-cp38-cp38-manylinux_2_24_x86_64.whl

root@2b11cc15c7f8:/mnt# wget -P x86_64 https://github.com/kendryte/nncase/releases/download/v1.8.0/nncase_k510-1.8.0.20220930-py2.py3-none-manylinux_2_24_x86_64.whl

root@2b11cc15c7f8:/mnt# pip3 install x86_64/*.whl
```



### 查看版本信息

```python
root@469e6a4a9e71:/mnt# python3
Python 3.8.10 (default, Jun  2 2021, 10:49:15)
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import _nncase
>>> print(_nncase.__version__)
1.8.0-55be52f
```



## nncase 编译模型APIs

### CompileOptions

#### 功能描述

CompileOptions类, 用于配置nncase编译选项

#### 类定义

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

各属性说明如下

| 属性名称         | 类型   | 是否必须 | 描述                                                         |
| ---------------- | ------ | -------- | ------------------------------------------------------------ |
| target           | string | 是       | 指定编译目标, 如'k210', 'k510'                               |
| quant_type       | string | 否       | 指定数据量化类型, 如'uint8', 'int8', 'int16'                 |
| w_quant_type     | string | 否       | 指定权重量化类型, 如'uint8', 'int8', 'int16', 默认为'uint8'  |
| use_mse_quant_w  | bool   | 否       | 指定权重量化时是否使用最小化均方误差(mean-square error, MSE)算法优化量化参数 |
| split_w_to_act   | bool   | 否       | 指定是否将权重数据平衡到激活数据中                           |
| preprocess       | bool   | 否       | 是否开启前处理，默认为False                                  |
| swapRB           | bool   | 否       | 是否交换RGB输入数据的红和蓝两个通道(RGB-->BGR或者BGR-->RGB)，默认为False |
| mean             | list   | 否       | 前处理标准化参数均值，默认为[0, 0, 0]                        |
| std              | list   | 否       | 前处理标准化参数方差，默认为[1, 1, 1]                        |
| input_range      | list   | 否       | 输入数据反量化后对应浮点数的范围，默认为[0，1]               |
| output_range     | list   | 否       | 输出定点数据前对应浮点数的范围，默认为空，使用模型实际浮点输出范围 |
| input_shape      | list   | 否       | 指定输入数据的shape，input_shape的layout需要与input layout保持一致，输入数据的input_shape与模型的input shape不一致时会进行letterbox操作(resize/pad等) |
| letterbox_value  | float  | 否       | 指定前处理letterbox的填充值                                  |
| input_type       | string | 否       | 指定输入数据的类型, 默认为'float32'                          |
| output_type      | string | 否       | 指定输出数据的类型, 如'float32', 'uint8'(仅用于指定量化情况下), 默认为'float32' |
| input_layout     | string | 否       | 指定输入数据的layout, 如'NCHW', 'NHWC'. 若输入数据layout与模型本身layout不同, nncase会插入transpose进行转换 |
| output_layout    | string | 否       | 指定输出数据的layout, 如'NCHW', 'NHWC'. 若输出数据layout与模型本身layout不同, nncase会插入transpose进行转换 |
| model_layout     | string | 否       | 指定模型的layout，默认为空，当tflite模型layout为‘NCHW’，Onnx和Caffe模型layout为‘NHWC’时需指定 |
| is_fpga          | bool   | 否       | 指定kmodel是否用于fpga, 默认为False                          |
| dump_ir          | bool   | 否       | 指定是否dump IR, 默认为False                                 |
| dump_asm         | bool   | 否       | 指定是否dump asm汇编文件, 默认为False                        |
| dump_quant_error | bool   | 否       | 指定是否dump量化前后的模型误差                               |
| dump_dir         | string | 否       | 前面指定dump_ir等开关后, 这里指定dump的目录, 默认为空字符串  |
| benchmark_only   | bool   | 否       | 指定kmodel是否只用于benchmark, 默认为False                   |

> 1. mean和std为浮点数进行normalize的参数，用户可以自由指定.
> 2. input range为浮点数的范围，即如果输入数据类型为uint8，则input range为反量化到浮点之后的范围（可以不为0~1），可以自由指定.
> 3. input_shape需要按照input_layout进行指定，以[1，224，224，3]为例，如果input_layout为NCHW，则input_shape需指定为[1,3,224,224];input_layout为NHWC，则input_shape需指定为[1,224,224,3];
>
> 例如:
>
> 1. 输入数据类型为uint8，range为0~255，input_range为0~255，则反量化的作用只是进行类型转化，将uint8的数据转化为float32，mean和std参数仍然可以按照0~255的数据进行指定.
> 2. 输入数据类型为uint8，range为0~255，input_range为0~1，则反量化会将定点数转化为浮点数0~1，mean和std参数需要按照0~1的数据进行指定。

#### 代码示例

实例化CompileOptions, 配置各属性的值

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

#### 功能描述

ImportOptions类, 用于配置nncase导入选项

#### 类定义

```python
py::class_<import_options>(m, "ImportOptions")
    .def(py::init())
    .def_readwrite("output_arrays", &import_options::output_arrays);
```

各属性说明如下

| 属性名称      | 类型   | 是否必须 | 描述     |
| ------------- | ------ | -------- | -------- |
| output_arrays | string | 否       | 输出名称 |

#### 代码示例

实例化ImportOptions, 配置各属性的值

```python
# import_options
import_options = nncase.ImportOptions()
import_options.output_arrays = 'output' # Your output node name
```

### PTQTensorOptions

#### 功能描述

PTQTensorOptions类, 用于配置nncase PTQ选项

#### 类定义

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

各属性说明如下

| 字段名称         | 类型   | 是否必须 | 描述                                                                                  |
| ---------------- | ------ | -------- | ------------------------------------------------------------------------------------- |
| calibrate_method | string | 否       | 校准方法 ,  支持'no_clip', 'l2', 'kld_m0', 'kld_m1', 'kld_m2', 'cdf', 默认是'no_clip' |
| samples_count    | int    | 否       | 样本个数                                                                              |

#### set_tensor_data()

##### 功能描述

设置校正数据

##### 接口定义

```python
set_tensor_data(calib_data)
```

##### 输入参数

| 参数名称   | 类型   | 是否必须 | 描述     |
| ---------- | ------ | -------- | -------- |
| calib_data | byte[] | 是       | 校正数据 |

##### 返回值

N/A

##### 代码示例

```python
# ptq_options
ptq_options = nncase.PTQTensorOptions()
ptq_options.samples_count = cfg.generate_calibs.batch_size
ptq_options.set_tensor_data(np.asarray([sample['data'] for sample in self.calibs]).tobytes())
```

### Compiler

#### 功能描述

Compiler类, 用于编译神经网络模型

#### 类定义

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

#### 代码示例

```python
compiler = nncase.Compiler(compile_options)
```

#### import_tflite()

##### 功能描述

导入tflite模型

##### 接口定义

```python
import_tflite(model_content, import_options)
```

##### 输入参数

| 参数名称       | 类型          | 是否必须 | 描述           |
| -------------- | ------------- | -------- | -------------- |
| model_content  | byte[]        | 是       | 读取的模型内容 |
| import_options | ImportOptions | 是       | 导入选项       |

##### 返回值

N/A

##### 代码示例

```python
model_content = read_model_file(model)
compiler.import_tflite(model_content, import_options)
```

#### import_onnx()

##### 功能描述

导入onnx模型

##### 接口定义

```python
import_onnx(model_content, import_options)
```

##### 输入参数

| 参数名称       | 类型          | 是否必须 | 描述           |
| -------------- | ------------- | -------- | -------------- |
| model_content  | byte[]        | 是       | 读取的模型内容 |
| import_options | ImportOptions | 是       | 导入选项       |

##### 返回值

N/A

##### 代码示例

```python
model_content = read_model_file(model)
compiler.import_onnx(model_content, import_options)
```

#### import_caffe()

##### 功能描述

导入caffe模型

> 用户需在本地机器自行编译/安装caffe.

##### 接口定义

```python
import_caffe(caffemodel, prototxt)
```

##### 输入参数

| 参数名称   | 类型   | 是否必须 | 描述                 |
| ---------- | ------ | -------- | -------------------- |
| caffemodel | byte[] | 是       | 读取的caffemodel内容 |
| prototxt   | byte[] | 是       | 读取的prototxt内容   |

##### 返回值

N/A

##### 代码示例

```python
# import
caffemodel = read_model_file('test.caffemodel')
prototxt = read_model_file('test.prototxt')
compiler.import_caffe(caffemodel, prototxt)
```

#### use_ptq()

##### 功能描述

设置PTQ配置选项

##### 接口定义

```python
use_ptq(ptq_options)
```

##### 输入参数

| 参数名称    | 类型             | 是否必须 | 描述        |
| ----------- | ---------------- | -------- | ----------- |
| ptq_options | PTQTensorOptions | 是       | PTQ配置选项 |

##### 返回值

N/A

##### 代码示例

```python
compiler.use_ptq(ptq_options)
```

#### compile()

##### 功能描述

编译神经网络模型

##### 接口定义

```python
compile()
```

##### 输入参数

N/A

##### 返回值

N/A

##### 代码示例

```python
compiler.compile()
```

#### gencode_tobytes()

##### 功能描述

生成代码字节流

##### 接口定义

```python
gencode_tobytes()
```

##### 输入参数

N/A

##### 返回值

bytes[]

##### 代码示例

```python
kmodel = compiler.gencode_tobytes()
with open(os.path.join(infer_dir, 'test.kmodel'), 'wb') as f:
    f.write(kmodel)
```

## 编译模型示例

### 编译float32 tflite模型

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

### 编译float32 onnx模型

针对onnx模型, 建议先使用[ONNX Simplifier](https://github.com/daquexian/onnx-simplifier)进行简化, 然后再使用nncase编译.

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

### 编译float32 caffe模型

caffe可去[kendryte caffe](https://github.com/kendryte/caffe/releases)取wheel包并安装进行测试.

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

### 编译添加前处理float32 tflite模型

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

### 编译uint8量化tflite模型

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

## 部署 nncase runtime

### K210

1. 从 [Release](https://github.com/kendryte/nncase/releases) 页面下载 `k210-runtime.zip`。
2. 解压到 [kendryte-standalone-sdk](https://github.com/kendryte/kendryte-standalone-sdk) 's `lib/nncase/v1` 目录。

## nncase 推理模型APIs

除了编译模型APIs, nncase还提供了推理模型的APIs, 在PC上可推理前面编译生成的kmodel,  用来验证nncase推理结果和相应深度学习框架的runtime的结果是否一致等.

### MemoryRange

#### 功能描述

MemoryRange类, 用于表示内存范围

#### 类定义

```python
py::class_<memory_range>(m, "MemoryRange")
    .def_readwrite("location", &memory_range::memory_location)
    .def_property(
        "dtype", [](const memory_range &range) { return to_dtype(range.datatype); },
        [](memory_range &range, py::object dtype) { range.datatype = from_dtype(py::dtype::from_args(dtype)); })
    .def_readwrite("start", &memory_range::start)
    .def_readwrite("size", &memory_range::size);
```

各属性说明如下

| 属性名称 | 类型           | 是否必须 | 描述                                                                       |
| -------- | -------------- | -------- | -------------------------------------------------------------------------- |
| location | int            | 否       | 内存位置, 0表示input, 1表示output, 2表示rdata, 3表示data, 4表示shared_data |
| dtype    | python数据类型 | 否       | 数据类型                                                                   |
| start    | int            | 否       | 内存起始地址                                                               |
| size     | int            | 否       | 内存大小                                                                   |

#### 代码示例

实例化MemoryRange

```python
mr = nncase.MemoryRange()
```

### RuntimeTensor

#### 功能描述

RuntimeTensor类, 用于表示运行时tensor

#### 类定义

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

各属性说明如下

| 属性名称 | 类型 | 是否必须 | 描述             |
| -------- | ---- | -------- | ---------------- |
| dtype    | int  | 否       | tensor的数据类型 |
| shape    | list | 否       | tensor的形状     |

#### from_numpy()

##### 功能描述

从numpy.ndarray构造RuntimeTensor对象

##### 接口定义

```python
from_numpy(py::array arr)
```

##### 输入参数

| 参数名称 | 类型          | 是否必须 | 描述              |
| -------- | ------------- | -------- | ----------------- |
| arr      | numpy.ndarray | 是       | numpy.ndarray对象 |

##### 返回值

RuntimeTensor

##### 代码示例

```python
tensor = nncase.RuntimeTensor.from_numpy(self.inputs[i]['data'])
```

#### copy_to()

##### 功能描述

拷贝RuntimeTensor

##### 接口定义

```python
copy_to(RuntimeTensor to)
```

##### 输入参数

| 参数名称 | 类型          | 是否必须 | 描述              |
| -------- | ------------- | -------- | ----------------- |
| to       | RuntimeTensor | 是       | RuntimeTensor对象 |

##### 返回值

N/A

##### 代码示例

```python
sim.get_output_tensor(i).copy_to(to)
```

#### to_numpy()

##### 功能描述

将RuntimeTensor转换为numpy.ndarray对象

##### 接口定义

```python
to_numpy()
```

##### 输入参数

N/A

##### 返回值

numpy.ndarray对象

##### 代码示例

```python
arr = sim.get_output_tensor(i).to_numpy()
```

### Simulator

#### 功能描述

Simulator类, 用于在PC上推理kmodel

#### 类定义

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

各属性说明如下

| 属性名称     | 类型 | 是否必须 | 描述     |
| ------------ | ---- | -------- | -------- |
| inputs_size  | int  | 否       | 输入个数 |
| outputs_size | int  | 否       | 输出个数 |

#### 代码示例

实例化Simulator

```python
sim = nncase.Simulator()
```

#### load_model()

##### 功能描述

加载kmodel

##### 接口定义

```python
load_model(model_content)
```

##### 输入参数

| 参数名称      | 类型   | 是否必须 | 描述         |
| ------------- | ------ | -------- | ------------ |
| model_content | byte[] | 是       | kmodel字节流 |

##### 返回值

N/A

##### 代码示例

```python
sim.load_model(kmodel)
```

#### get_input_desc()

##### 功能描述

获取指定索引的输入的描述信息

##### 接口定义

```python
get_input_desc(index)
```

##### 输入参数

| 参数名称 | 类型 | 是否必须 | 描述       |
| -------- | ---- | -------- | ---------- |
| index    | int  | 是       | 输入的索引 |

##### 返回值

MemoryRange

##### 代码示例

```python
input_desc_0 = sim.get_input_desc(0)
```

#### get_output_desc()

##### 功能描述

获取指定索引的输出的描述信息

##### 接口定义

```python
get_output_desc(index)
```

##### 输入参数

| 参数名称 | 类型 | 是否必须 | 描述       |
| -------- | ---- | -------- | ---------- |
| index    | int  | 是       | 输出的索引 |

##### 返回值

MemoryRange

##### 代码示例

```python
output_desc_0 = sim.get_output_desc(0)
```

#### get_input_tensor()

##### 功能描述

获取指定索引的输入的RuntimeTensor

##### 接口定义

```python
get_input_tensor(index)
```

##### 输入参数

| 参数名称 | 类型 | 是否必须 | 描述             |
| -------- | ---- | -------- | ---------------- |
| index    | int  | 是       | 输入tensor的索引 |

##### 返回值

RuntimeTensor

##### 代码示例

```python
input_tensor_0 = sim.get_input_tensor(0)
```

#### set_input_tensor()

##### 功能描述

设置指定索引的输入的RuntimeTensor

##### 接口定义

```python
set_input_tensor(index, tensor)
```

##### 输入参数

| 参数名称 | 类型          | 是否必须 | 描述                    |
| -------- | ------------- | -------- | ----------------------- |
| index    | int           | 是       | 输入RuntimeTensor的索引 |
| tensor   | RuntimeTensor | 是       | 输入RuntimeTensor       |

##### 返回值

N/A

##### 代码示例

```python
sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(self.inputs[0]['data']))
```

#### get_output_tensor()

##### 功能描述

获取指定索引的输出的RuntimeTensor

##### 接口定义

```python
get_output_tensor(index)
```

##### 输入参数

| 参数名称 | 类型 | 是否必须 | 描述                    |
| -------- | ---- | -------- | ----------------------- |
| index    | int  | 是       | 输出RuntimeTensor的索引 |

##### 返回值

RuntimeTensor

##### 代码示例

```python
output_arr_0 = sim.get_output_tensor(0).to_numpy()
```

#### set_output_tensor()

##### 功能描述

设置指定索引的输出的RuntimeTensor

##### 接口定义

```python
set_output_tensor(index, tensor)
```

##### 输入参数

| 参数名称 | 类型          | 是否必须 | 描述                    |
| -------- | ------------- | -------- | ----------------------- |
| index    | int           | 是       | 输出RuntimeTensor的索引 |
| tensor   | RuntimeTensor | 是       | 输出RuntimeTensor       |

##### 返回值

N/A

##### 代码示例

```python
sim.set_output_tensor(0, tensor)
```

#### run()

##### 功能描述

运行kmodel推理

##### 接口定义

```python
run()
```

##### 输入参数

N/A

##### 返回值

N/A

##### 代码示例

```python
sim.run()
```

# ncc

## 命令行

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

## 描述

`ncc` 是 nncase 的命令行工具。它有两个命令： `compile` 和 `infer`。

`compile` 命令将你训练好的模型 (`.tflite`, `.caffemodel`, `.onnx`) 编译到 `.kmodel`。

- `-i, --input-format` 用来指定输入模型的格式。nncase 现在支持 `tflite`、`caffe` 和 `onnx` 输入格式。
- `-t, --target` 用来指定你想要你的模型在哪种目标设备上运行。`cpu` 几乎所有平台都支持的通用目标。`k210` 是 Kendryte K210 SoC 平台。如果你指定了 `k210`，这个模型就只能在 K210 运行或在你的 PC 上模拟运行。
- `<input file>` 用于指定输入模型文件
- `--input-prototxt`用于指定caffe模型的prototxt文件
- `<output file>` 用于指定输出模型文件
- `--output-arrays `用于指定输出结点的名称
- `--quant-type` 用于指定数据的量化类型, 如 `uint8`/`int8`/`int16, 默认是`uint8
- `--w-quant-type` 用于指定权重的量化类型, 如 `uint8`/`int8`/`int16, 默认是`uint8
- `--use-mse-quant-w`指定是否使用最小化mse(mean-square error, 均方误差)算法来量化权重.
- `--dataset` 用于提供量化校准集来量化你的模型。你需要从训练集中选择几百到上千个数据放到这个目录里。
- `--dataset-format` 用于指定量化校准集的格式。默认是 `image`，nncase 将使用 `opencv` 读取你的图片，并自动缩放到你的模型输入需要的尺寸。如果你的输入有 3 个通道，ncc 会将你的图片转换为值域是 [0,1] 布局是 `NCHW` 的张量。如果你的输入只有 1 个通道，ncc 会灰度化你的图片。如果你的数据集不是图片（例如音频或者矩阵），把它设置为 `raw`。这种场景下你需要把你的数据集转换为 float 张量的二进制文件。
- `--dump-range-dataset` 用于提供统计范围数据集来统计原始模型每个节点输出数据范围。你需要从训练集中选择几百到上千个数据放到这个目录里。
- `--dump-range-dataset-format` 用于指定统计范围数据集的格式。默认是 `image`，nncase 将使用 `opencv` 读取你的图片，并自动缩放到你的模型输入需要的尺寸。如果你的输入有 3 个通道，ncc 会将你的图片转换为值域是 [0,1] 布局是 `NCHW` 的张量。如果你的输入只有 1 个通道，ncc 会灰度化你的图片。如果你的数据集不是图片（例如音频或者矩阵），把它设置为 `raw`。这种场景下你需要把你的数据集转换为 float 张量的二进制文件。
- `--calibrate-method` 用于设置量化校准方法，它被用来选择最优的激活函数值域。默认值是 `no_clip`，ncc 会使用整个激活函数值域。如果你需要更好的量化结果，你可以使用 `l2`，但它需要花更长的时间寻找最优值域。
- `--preprocess`指定是否预处理, 添加后表示开启预处理
- `--swapRB`指定**预处理时**是否交换红和蓝两个通道数据, 用于实现RGB2BGR或BGR2RGB功能
- `--mean`指定**预处理时**标准化参数均值,例如添加 `--mean "0.1 2.3 33.1f"`用于设置三个通道的均值.
- `--std`指定**预处理时**标准化参数方差,例如添加 `--std "1. 2. 3."`用于设置三个通道的方差.
- `--input-range`指定输入数据反量化后的数据范围,例如添加 `--input-range "0.1 2."`设置反量化的范围为 `[0.1~2]`.
- `--input-shape`指定输入数据的形状. 若与模型的输入形状不同, 则预处理时会做resize/pad等处理, 例如添加 `--input-shape "1 1 28 28"`指明当前输入图像尺寸.
- `--letterbox-value`用于指定预处理时pad填充的值.
- `--input-type` 用于指定推理时输入的数据类型。如果 `--input-type` 是 `uint8`，推理时你需要提供 RGB888 uint8 张量。如果 `--input-type` 是 `float`，你则需要提供 RGB float 张量.
- `--output-type` 用于指定推理时输出的数据类型。如 `float`/`uint8`,  `uint8`仅在量化模型时才有效. 默认是 `float`
- `--input-layout`用于指定输入数据的layout. 若输入数据的layout与模型的layout不同, 预处理会添加transpose进行转换.
- `--output-layout`用于指定输出数据的layout
- `--tcu-num`用于指定tcu个数, 默认值为0, 表示不配置tcu个数.
- `--is-fpga`指定编译后的kmodel是否运行在fpga上
- `--dump-ir` 是一个调试选项。当它打开时 ncc 会在工作目录产生一些 `.dot` 文件。你可以使用 `Graphviz` 或 [Graphviz Online](https://dreampuf.github.io/GraphvizOnline) 来查看这些文件。
- `--dump-asm` 是一个调试选项。当它打开时 ncc 会生成硬件指令文件compile.text.asm
- `--dump-quant-error`是一个调试选项, 用于dump量化错误信息
- `--dump-import-op-range`是一个调试选项, 用于dump import之后节点的数据范围，需要同时指定dump-range-dataset
- `--dump-dir`是一个调试选项, 用于指定dump目录.
- `--benchmark-only`是一个调试选项, 用于指定编译后的kmodel用于benchmark.

`infer` 命令可以运行你的 kmodel，通常它被用来调试。ncc 会将你模型的输出张量按 `NCHW` 布局保存到 `.bin` 文件。

- `<input file>` kmodel 的路径。
- `<output path>` ncc 输出目录。
- `--dataset` 测试集路径。
- `--dataset-format`和 `--input-layout`同 `compile` 命令中的含义。
