# 常见问题

## 1. 安装 `whl`包出错

### 1.1 `xxx.whl is not a supported wheel on this platform.`

A：升级 pip >= 20.3

```shell
pip install --upgrade pip
```

---

## 2.编译模型时报错

### 2.1 编译模型报错“System.NotSupportedException: Not Supported *** op: XXX”。

A：该异常表明 `XXX`算子尚未支持，可以在[nncase Github Issue](https://github.com/kendryte/nncase/issues)中提需求。当前目录下 `***_ops.md`文档，可以查看各个推理框架中已经支持的算子。

如果 `XXX`属于 `FAKE_QUANT`、`DEQUANTIZE`、`QUANTIZE`等量化相关的算子，表明当前模型属于量化模型，`nncase`目前不支持这类模型，请使用浮点模型来编译 `kmodel`。

### 2.2 "The configured user limit (128) on the number of inotify instances has been reached, or the per-process limit on the number of open file descriptors has been reached"。

A：使用 `sudo gedit /proc/sys/fs/inotify/max_user_instances`修改128为更大的值即可。

### 2.3 `RuntimeError: Failed to initialize hostfxr`

A：需要安装dotnet-sdk-7.0
- Linux:

    ```shell
    sudo apt-get update
    sudo apt-get install dotnet-sdk-7.0
    ```

- Windows: 请自行查阅微软官方文档。

### 2.4 "KeyNotFoundException: The given key 'K230' was not present in the dictionary"

A：需要安装nncase-kpu
- Linux：使用pip安装nncase-kpu `pip install nncase-kpu`
- Windows：在[nncase github tags界面](https://github.com/kendryte/nncase/tags)下载对应版本的whl包，然后使用pip安装。

>安装nncase-kpu之前，请先检查nncase版本，然后安装与nncase版本一致的nncase-kpu。

```shell
> pip show nncase | grep "Version:"
 Version: 2.8.0
(Linux)  > pip install nncase-kpu==2.8.0
(Windows)> pip install nncase_kpu-2.8.0-py2.py3-none-win_amd64.whl
```

### 2.5 `RuntimeError: Failed to get hostfxr path.`

A: 配置`dotnet`环境变量。 [dotnet issue #79237](https://github.com/dotnet/runtime/issues/79237)

```shell
export DOTNET_ROOT=/usr/share/dotnet
```

---

## 3. 推理时报错

### 3.1 推理的时候出现 `nncase.simulator.k230.sc: not found`。

或者以下情况：
- `"nncase.simulator.k230.sc: Permision denied."`
- `"Input/output error."`

A：将nncase的安装路径加入到 `PATH`环境变量中，同时检查一下`nncase`和`nncase-kpu`版本是否一致。

```shell
root@a52f1cacf581:/mnt# pip list | grep nncase
nncase                       2.1.1.20230721
nncase-kpu                   2.1.1.20230721
```

如果不一致，请安装相同版本的Python包 `pip install nncase==x.x.x.x nncase-kpu==x.x.x.x`。

---

## 4. k230开发板推理时报错

### 4.1 `data.size_bytes() == size = false (bool)`

A：以上这种情况通常有是app推理时的输入数据文件有错误，与模型输入shape不匹配或者与模型输入type不匹配。尤其当配置了前处理时需要检查这两个属性，添加前处理操作后，模型中增加了相关的节点，输入节点也会发生变化。如果 `input_shape`、`input_type`和原始模型不同，则需要以新配置的 `shape`，`type`为准来生成输入数据。

### 4.2 抛出 `std::bad_alloc`异常

A：通常是因为内存分配失败导致的，可做如下排查。

- 检查生成的kmodel是否超过当前系统可用内存
- 检查App是否存在内存泄露

### 4.3 加载模型时抛出异常

加载`kmodel`代码如下时，抛出异常 `terminate:Invalid kmodel`。

```CPP
interp.load_model(ifs).expect("Invalid kmodel");
```

A：是由于编译`kmodel`时的nncase版本与当前SDK版本不匹配导致，请按照[SDK、nncase版本对应关系](https://developer.canaan-creative.com/k230/dev/zh/03_other/K230_SDK_nncase%E7%89%88%E6%9C%AC%E5%AF%B9%E5%BA%94%E5%85%B3%E7%B3%BB.html)查询，并按照[更新nncase运行时库教程](https://developer.canaan-creative.com/k230/dev/zh/03_other/K230_SDK%E6%9B%B4%E6%96%B0nncase%E8%BF%90%E8%A1%8C%E6%97%B6%E5%BA%93%E6%8C%87%E5%8D%97.html)解决。