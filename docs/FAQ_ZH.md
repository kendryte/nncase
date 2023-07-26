## 常见问题

### 安装 `whl`包出错

Q：`xxx.whl is not a supported wheel on this platform.`

A： 升级 pip >= 20.3 `pip install --upgrade pip`

### 编译模型时报错

#### 1. `System.NotSupportedException`

Q：编译模型报错"System.NotSupportedException: Not Supported TFLite op: FAKE_QUANT"

A：该异常表明存在算子尚未支持，可以查看当前目录下 `*_ops.md`中已经支持的算子。如果模型本身是量化过的，会存在例如 `FAKE_QUANT`，`DEQUANTIZE`，`QUANTIZE`，这一类算子目前不支持，需要用户使用非量化模型编译kmodel

#### 2. `System.IO.IOException`

Q: 下载nncase仓库自己编译后运行test出现这个错误，"The configured user limit (128) on the number of inotify instances has been reached, or the per-process limit on the number of open file descriptors has been reached".

A：使用 `sudo gedit /proc/sys/fs/inotify/max_user_instances`修改128为更大的值即可

### 推理时报错

#### 1. `nncase.simulator.k230.sc: not found `

Q：在编译kmodel正常， 但是推理的时候出现"nncase.simulator.k230.sc: not found"的错误

A：需要检查nncase和nncase-kpu的版本是否一致

```shell
root@a52f1cacf581:/mnt# pip list | grep nncase
nncase                       2.1.1.20230721
nncase-kpu                   2.1.1.20230721
```

如果不一致，请安装相同版本的Python包 `pip install nncase==x.x.x.x nncase-kpu==x.x.x.x`

### k230开发板推理时报错

#### Q1. `data.size_bytes() == size = false (bool)`

A：以上这种情况通常有是app推理时的输入数据文件有错误，与模型输入shape不匹配或者与模型输入type不匹配。尤其当配置了前处理时需要检查这两个属性，添加前处理操作后，模型中增加了相关的节点，输入节点也会发生变化。如果 `input_shape`、`input_type`和原始模型不同，则需要以新配置的 `shape`，`type`为准来生成输入数据。

#### Q2.抛出 `std::bad_alloc`异常

A：通常是因为内存分配失败导致的, 可做如下排查.

- 检查生成的kmodel是否超过当前系统可用内存
- 检查App是否存在内存泄露
