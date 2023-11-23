# ShapeBucket使用说明

ShapeBucket是针对动态shape的一种解决方案，会根据输入长度的范围以及指定的段的数量来对动态shape进行优化。该功能默认为false，需要打开对应的option才能生效，除了指定对应的字段信息，其他流程与编译静态模型没有区别。

对应的不同CompileOptions中的字段

| 字段名称                        | 类型                    | 是否必须 | 描述                                             |
|-----------------------------|-----------------------|------|------------------------------------------------|
| shape_bucket_enable         | bool                  | 是    | 是否开启ShapeBucket功能，默认为False。在 `dump_ir=True`时生效 |
| shape_bucket_range_info     | Dict[str, [int, int]] | 是    | 每个输入shape维度信息中的变量的范围，最小值必须大于等于1                |
| shape_bucket_segments_count | int                   | 是    | 输入变量的范围划分为几段                                   |
| shape_bucket_fix_var_map    | Dict[str, int]        | 否    | 固定shape维度信息中的变量为特定的值                           |

## onnx

在模型的shape中会有些维度为变量名字，这里以一个onnx模型的输入为例

> tokens: int64[batch_size, tgt_seq_len]
>
> step: float32[seq_len, batch_size]

对应这个输入有如下的配置

```python
shape_bucket_options = nncase.ShapeBucketOptions()
shape_bucket_options.shape_bucket_enable = True
shape_bucket_options.shape_bucket_range_info = {"seq_len": [1, 100], "tgt_seq_len": [1, 100]}
shape_bucket_options.shape_bucket_segments_count = 2
shape_bucket_options.shape_bucket_fix_var_map = {"batch_size": 3}
```

shape的维度信息中存在seq_len，tgt_seq_len，batch_size这三个变量。首先是batch_size，虽然是变量的但实际应用的时候固定为3，因此在
**fix_var_map**中添加batch_size = 3，在运行的时候会将这个维度固定为3。

seq_len，tgt_seq_len两个是实际会发生改变的，因此需要配置这两个变量的实际范围，也就是**range_info**的信息。**segments_count**
是实际分段的数量，会根据范围等分为几份，对应的编译时间也会相应增加几倍。

## tflite

tflite的模型与onnx不同，shape上暂未标注维度的名称，目前只支持输入中具有一个维度是动态的，并且名称统一配置为-1，配置方式如下

```cpp
shape_bucket_options = nncase.ShapeBucketOptions()
shape_bucket_options.shape_bucket_enable = True
shape_bucket_options.shape_bucket_range_info = {"-1":[1, 100]}
shape_bucket_options.shape_bucket_segments_count = 2
shape_bucket_options.shape_bucket_fix_var_map = {"batch_size" : 3}
```

配置完这些选项后整个编译的流程和静态shape一致。

# ShapeBucket's operation instructions

ShapeBucket is a solution for dynamic shapes that optimizes them based on the range of input lengths and the number of
specified segments. This function defaults to false and requires opening the corresponding option to take effect. Apart
from specifying the corresponding field information, the other processes are no different from compiling a static model.

Corresponding fields in different CompileOptions.

| Field name                  | datatype              | necessary or not | description                                                                                                           |
|-----------------------------|-----------------------|------------------|-----------------------------------------------------------------------------------------------------------------------|
| shape_bucket_enable         | bool                  | yes              | Whether to enable the ShapeBucket function, default to False. It will work when `dump_ir=True`.                       |
| shape_bucket_range_info     | Dict[str, [int, int]] | yes              | The range of variables in each input shape dimension information must have a minimum value greater than or equal to 1 |
| shape_bucket_segments_count | int                   | yes              | The range of input variables is divided into several segments.                                                        |
| shape_bucket_fix_var_map    | Dict[str, int]        | no               | The variables in the fixed shape dimension information are specific values.                                           |

## onnx

In the shape of the model, there will be some dimensions with variable names, taking the input of an ONNX model as an
example

> tokens: int64[batch_size, tgt_seq_len]
>
> step: float32[seq_len, batch_size]

The corresponding configuration for this input is as follows:

```python
shape_bucket_options = nncase.ShapeBucketOptions()
shape_bucket_options.shape_bucket_enable = True
shape_bucket_options.shape_bucket_range_info = {"seq_len": [1, 100], "tgt_seq_len": [1, 100]}
shape_bucket_options.shape_bucket_segments_count = 2
shape_bucket_options.shape_bucket_fix_var_map = {"batch_size": 3}
```

There is seq in the dimensional information of the shape_len, tgt_seq_len, batch_size.
Firstly, batch_size, although it is a variable, is fixed to 3 in practical application, so add batch_size = 3 in *
*fix_var_map**,
this dimension will be fixed to 3 during runtime.

seq_len，tgt_seq_len两个是实际会发生改变的，因此需要配置这两个变量的实际范围，也就是**range_info**的信息。**segments_count**
是实际分段的数量，会根据范围等分为几份，对应的编译时间也会相应增加几倍。

seq_len and tgt_seq_len will actually change, so it is necessary to configure the actual range of these two variables,
which is the information of **range_info**.**segments_count** is the actual number of segments, which will be divided
into several equal parts based on the scope, and the corresponding compilation time will also increase several times.

## tflite

The model of tflite is different from onnx. The shape does not currently have a dimension name labeled. Currently, only
one dimension in the input is supported to be dynamic, and the name is uniformly configured as -1. The configuration
method is as follows.

```cpp
shape_bucket_options = nncase.ShapeBucketOptions()
shape_bucket_options.shape_bucket_enable = True
shape_bucket_options.shape_bucket_range_info = {"-1":[1, 100]}
shape_bucket_options.shape_bucket_segments_count = 2
shape_bucket_options.shape_bucket_fix_var_map = {"batch_size" : 3}
```

After configuring these options, the entire compilation process is consistent with the static shape.