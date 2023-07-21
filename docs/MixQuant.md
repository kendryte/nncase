# 混合量化使用说明

**混合量化需要在量化功能开启状态才可以使用，需要保证编译脚本中存在以下设置开启量化功能。**

```python
compile_options = nncase.CompileOptions()
#compile_options.xxx = xxx
compiler = nncase.Compiler(compile_options)
ptq_options = nncase.PTQTensorOptions()
#ptq_options.xxx = xxx
...
compiler.use_ptq(ptq_options)
```

## 1. 混合量化相关的功能选项说明

```python
ptq_options.quant_scheme = ""
ptq_options.export_quant_scheme = False
ptq_options.export_weight_range_by_channel = False
```

* **quant_scheme：导入量化参数配置文件的路径**
* **export_quant_scheme：是否导出量化参数配置文件**
* **export_weight_range_by_channel：是否导出** `bychannel`形式的weights量化参数，为了保证量化效果，该参数建议设置为 `True`

---

## 2. 导出量化参数配置文件

*由于量化参数配置文件输入编译中间信息，需要用户打开 `dump_ir`功能*。

```python
compile_options.dump_ir = True
```

```python
ptq_options.quant_scheme = ""
ptq_options.export_quant_scheme = True
ptq_options.export_weight_range_by_channel = True
```

**生成的量化参数配置文件与** `kmodel`同目录

---

## 3. 修改量化参数配置文件

**量化参数配置文件内格式如下：**

```json
{
  "Version": "1.0",
  "Model": null,
  "Outputs": [
    {
      "Name": "input",
      "DataType": "u8",
      "DataRange": [
        {
          "Min": 0.0,
          "Max": 0.99999493,
          "IsFull": false
        }
      ],
      "DataRangeMode": "by_tensor"
    },
    {
      "Name": "Const_90",
      "DataType": "i16",
      "DataRange": [
        {
          "Min": -0.82569844,
          "Max": 1.2145407,
          "IsFull": false
        },
        {
          "Min": -3.459431,
          "Max": 2.9377983,
          "IsFull": false
        },
        //...
      ],
      "DataRangeMode": "by_channel"
    },
    {
      "Name": "MobilenetV2/Conv/Relu6",
      "DataType": "f16",
      "DataRange": [
        {
          "Min": 0.0,
          "Max": 6.0,
          "IsFull": false
        }
      ],
      "DataRangeMode": "by_tensor"
    },
    // ...
  ]
}
```

**可以自行修改每层的** `Min` ,`Max`,`DataType`来调整量化的效果，其中 `DataType` 支持设置为 `u8` ,`i8`, `i16`, `f16`, `f32`

## 4. 导入量化参数配置文件

**需要在编译脚本中设置量化参数配置文件**

```python
ptq_options.quant_scheme = "./QuantScheme.json" # path to your 'QuantScheme.json'
ptq_options.export_quant_scheme = False
ptq_options.export_weight_range_by_channel = False # whatever
```

**在导入量化参数配置文件后不需要额外设置校正集**

```python
ptq_options.set_tensor_data([])
```
