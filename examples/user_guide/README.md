模型编译推理参考Jupyter脚本：[User_guide](./simulate.ipynb)，脚本中包含了单输入和多输入的示例。也可以使用单独的编译脚本 [Single build](../../docs/USAGE_ZH.md#编译模型示例)完成kmodel的编译。

如果在Docker中运行Jupyter脚本，可以参考[配置Jupyter lab](https://github.com/kunjing96/docker-jupyterlab#32-%E9%85%8D%E7%BD%AEjupyter-lab)进行配置。

在执行脚本之前需要根据自身需求修改以下内容：

1. `compile_kmodel`函数中 `compile_options`,`ptq_options`相关信息
   `compile_options`详细信息见[CompileOptions](../../docs/USAGE_ZH.md#CompileOptions)
   `ptq_options`详细信息见[PTQTensorOptions](../../docs/USAGE_ZH.md#PTQTensorOptions)
2. `compile kmodel single input(multiple inputs)`部分
   修改 `model_path`和 `dump_path`，用于指定模型路径和编译期间文件生成路径。
   修改 `calib_data`的实现，数据格式见注释。
3. `run kmodel(simulate)`部分，修改 `input_data`的实现，数据格式见注释。

推理结束后，会在 `dump_path`路径下生成 `kmodel`、输出结果和编译期间的文件。