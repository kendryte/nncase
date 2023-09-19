# 常见问题

## 1. 安装 `whl`包出错

### 1.1 Q：`xxx.whl is not a supported wheel on this platform.`

A：升级 pip >= 20.3 `pip install --upgrade pip`

---

## 2.编译模型时报错

### 2.1 `System.NotSupportedException`

#### 2.1.1 Q：编译模型报错“System.NotSupportedException: Not Supported *** op: XXX”。

A：该异常表明 `XXX`算子尚未支持，可以在[nncase Github Issue](https://github.com/kendryte/nncase/issues)中提需求。当前目录下 `***_ops.md`文档，可以查看各个推理框架中已经支持的算子。

如果 `XXX`属于 `FAKE_QUANT`、`DEQUANTIZE`、`QUANTIZE`等量化相关的算子，表明当前模型属于量化模型，`nncase`目前不支持这类模型，请使用浮点模型来编译 `kmodel`。

### 2.2 `System.IO.IOException`

#### 2.2.1 Q：下载 `nncase`仓库自己编译后，运行test出现这个错误"The configured user limit (128) on the number of inotify instances has been reached, or the per-process limit on the number of open file descriptors has been reached"。

A1：使用 `sudo gedit /proc/sys/fs/inotify/max_user_instances`修改128为更大的值即可。

### 2.3 `initialize`相关
#### 2.3.1 Q：编译模型出现如下错误`RuntimeError: Failed to initialize hostfxr`
A1：需要安装dotnet-7.0

---

## 3. 推理时报错

### 3.1 Q：在编译kmodel正常， 但是推理的时候出现 `nncase.simulator.k230.sc: not found`的错误。

A：将nncase的安装路径加入到 `PATH`环境变量中，同时检查一下nncase和nncase-kpu版本是否一致。

```shell
root@a52f1cacf581:/mnt# pip list | grep nncase
nncase                       2.1.1.20230721
nncase-kpu                   2.1.1.20230721
```

如果不一致，请安装相同版本的Python包 `pip install nncase==x.x.x.x nncase-kpu==x.x.x.x`。

---

## 4. k230开发板推理时报错

### 4.1 Q：`data.size_bytes() == size = false (bool)`

A：以上这种情况通常有是app推理时的输入数据文件有错误，与模型输入shape不匹配或者与模型输入type不匹配。尤其当配置了前处理时需要检查这两个属性，添加前处理操作后，模型中增加了相关的节点，输入节点也会发生变化。如果 `input_shape`、`input_type`和原始模型不同，则需要以新配置的 `shape`，`type`为准来生成输入数据。

### 4.2 Q：抛出 `std::bad_alloc`异常

A：通常是因为内存分配失败导致的，可做如下排查。

- 检查生成的kmodel是否超过当前系统可用内存
- 检查App是否存在内存泄露
