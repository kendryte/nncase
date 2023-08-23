# FAQ

[toc]

## 1. Error installing `whl` package

### 1.1 Q: `xxx.whl is not a supported wheel on this platform`

A: Upgrade pip >= 20.3 using `pip install --upgrade pip`

---

## 2. Compile-time errors

### 2.1 "System.NotSupportedException"

#### 2.1.1 Q: Compile model reported error "System.NotSupportedException: Not Supported *** op: XXX"

A: This exception indicates that there are operators, `XXX`, that are not yet supported. You can create a issue in [nncase Github Issue](https://github.com/kendryte/nncase/issues). In the current directory `***_ops.md`, you can view the operators already supported in each inference framework.

If 'XXX' belongs to quantization-related operators such as `FAKE_QUANT`, `DEQUANTIZE`, `QUANTIZE`, it indicates that the current model is a quantized model, and 'nncase' does not currently support such models, please compile `kmodel` using a floating point model.

### 2.2 "System.IO.IOException"

#### 2.2.1 Q: Downloading the `nncase` repository and compiling it yourself and running test gives this error, `The configured user limit (128) on the number of inotify instances has been reached, or the per-process limit on the number of open file descriptors has been reached`.

A: Use `sudo gedit /proc/sys/fs/inotify/max_user_instances` to change 128 to a larger value.

### 2.3 `initialize` error

#### 2.3.1 Q："RuntimeError: Failed to initialize hostfxr" appears when compiling the kmodel.

A1：Need to install dotnet-7.0.

---

## 3. Runtime errors

### 3.1 Q: Compiling `kmodel` is fine, but when inferring, the error `nncase.simulator.k230.sc: not found`occurs.

A: First, make sure that the path of the nncase installation is added to the PATH environment variable. You need to check whether the versions of `nncase` and `nncase-kpu` are the same.

```shell
root@a52f1cacf581:/mnt# pip list | grep nncase
nncase 2.1.1.20230721
nncase-kpu 2.1.1.20230721
```

If inconsistent, install the same version of the Python package `pip install nncase==x.x.x.x nncase-kpu==x.x.x.x`

---

## 4. Runtime error on k230 development board

### 4.1 Q: `data.size_bytes() == size = false (bool)`

A: The above situation is usually caused by an error in the input data file of the app inference, which does not match the model input shape or the model input type. Especially when pre-processing is configured, you need to check ` input_shape` and `input_type ` of input data, after adding pre-processing operation, relevant nodes are added to the model, and the input node will also be changed. If   `input_shape `, `input_type `are different from the original model, the newly configured `shape `, `type` should be used to generate input data.

### 4.2 Q: `std::bad_alloc`

A: Usually it is caused by memory allocation failure, you can do the following troubleshooting.

- Check whether the generated `kmodel` exceeds the current available memory.
- Check whether the generated `kmodel` exceeds the current available system memory.
