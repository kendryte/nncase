# FAQ

[toc]

## 1. Error installing `whl` package

### 1.1 `xxx.whl is not a supported wheel on this platform`

A: Upgrade pip >= 20.3.

```shell
pip install --upgrade pip
```

---

## 2. Compile-time errors

### 2.1 Compile model reported error "System.NotSupportedException: Not Supported *** op: XXX"

A: This exception indicates that there are operators, `XXX`, that are not yet supported. You can create a issue in [nncase Github Issue](https://github.com/kendryte/nncase/issues). In the current directory `***_ops.md`, you can view the operators already supported in each inference framework.

If 'XXX' belongs to quantization-related operators such as `FAKE_QUANT`, `DEQUANTIZE`, `QUANTIZE`, it indicates that the current model is a quantized model, and 'nncase' does not currently support such models, please compile `kmodel` using a floating point model.

### 2.2 "The configured user limit (128) on the number of inotify instances has been reached, or the per-process limit on the number of open file descriptors has been reached."

A: Use `sudo gedit /proc/sys/fs/inotify/max_user_instances` to change 128 to a larger value.

### 2.3 `RuntimeError: Failed to initialize hostfxr`

A：Need to install dotnet-sdk-7.0.

- Linux:

    ```shell
    sudo apt-get update
    sudo apt-get install dotnet-sdk-7.0
    ```

- Windows: Refer to MicroSoft official website.

### 2.4 "KeyNotFoundException: The given key 'K230' was not present in the dictionary"

A: Need to install `nncase-kpu`.
- Linux: `pip install nncase-kpu`
- Windows: Sorry for that you need to download the `whl` package in [nncase github repo](https://github.com/kendryte/nncase/tags) and install it manually.

> Before install `nncase`, please make sure that the version of `nncase` is consistent with the version of `nncase-kpu`.

```shell
> pip show nncase | grep "Version:"
 Version: 2.8.0
(Linux)  > pip install nncase-kpu==2.8.0
(Windows)> pip install nncase_kpu-2.8.0-py2.py3-none-win_amd64.whl
```

### 2.5 `RuntimeError: Failed to get hostfxr path.`

A: Set `dotnet` enviroment. [dotnet issue #79237](https://github.com/dotnet/runtime/issues/79237)

```shell
export DOTNET_ROOT=/usr/share/dotnet
```

---

## 3. Runtime errors

### 3.1 When inferring, the error `nncase.simulator.k230.sc: not found` occurs.

Or these situations:

- `"nncase.simulator.k230.sc: Permision denied."`
- `"Input/output error."`

A: Make sure that the path of the nncase installation is added to the `PATH` environment variable. You need to check whether the versions of `nncase` and `nncase-kpu` are the same.

```shell
root@a52f1cacf581:/mnt# pip list | grep nncase
nncase 2.1.1.20230721
nncase-kpu 2.1.1.20230721
```

If inconsistent, install the same version of the Python package `pip install nncase==x.x.x.x nncase-kpu==x.x.x.x`

---

## 4. Runtime error on k230 development board

### 4.1 `data.size_bytes() == size = false (bool)`

A: The above situation is usually caused by an error in the input data file of the app inference, which does not match the model input shape or the model input type. Especially when pre-processing is configured, you need to check ` input_shape` and `input_type ` of input data, after adding pre-processing operation, relevant nodes are added to the model, and the input node will also be changed. If   `input_shape `, `input_type `are different from the original model, the newly configured `shape `, `type` should be used to generate input data.

### 4.2 `std::bad_alloc`

A: Usually it is caused by memory allocation failure, you can do the following troubleshooting.

- Check whether the generated `kmodel` exceeds the currently available system memory.
- Check App for memory leaks.

### 4.3 throw error when load model

The exception `terminate: Invalid kmodel` is thrown when attempting to load a `kmodel` as bellow.

```CPP
interp.load_model(ifs).expect("Invalid kmodel");
```

A：The issue arises due to a mismatch between the nncase version used when compiling the kmodel and the current SDK version. Please refer to the [SDK-nncase Version Correspondence](https://developer.canaan-creative.com/k230/dev/zh/03_other/K230_SDK_nncase%E7%89%88%E6%9C%AC%E5%AF%B9%E5%BA%94%E5%85%B3%E7%B3%BB.html) for a lookup, and follow the [Update the nncase Runtime Library Guide](https://developer.canaan-creative.com/k230/dev/zh/03_other/K230_SDK%E6%9B%B4%E6%96%B0nncase%E8%BF%90%E8%A1%8C%E6%97%B6%E5%BA%93%E6%8C%87%E5%8D%97.html) to resolve the problem.

