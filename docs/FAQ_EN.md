## FAQ

### Error installing `whl` package

Q: `xxx.whl is not a supported wheel on this platform.

A： Upgrade pip >= 20.3 using `pip install --upgrade pip`

### Error compiling model

#### 1. `System.NotSupportedException`

Q：Compile model reported error "System.NotSupportedException: Not Supported TFLite op: FAKE_QUANT"

A: This exception indicates that there are operators that are not yet supported. You can check the supported operators in `*_ops.md` in the current directory. If the model itself is quantized, there are operators such as `FAKE_QUANT`, `DEQUANTIZE`, `QUANTIZE`, which are not currently supported and require the user to compile the kmodel with a non-quantized model.

#### 2. `System.IO.IOException`

Q: Downloading the nncase repository and compiling it yourself and running test gives this error, "The configured user limit (128) on the number of inotify instances has been reached, or the per-process limit on the number of open file descriptors has been reached".

A: Use `sudo gedit /proc/sys/fs/inotify/max_user_instances` to change 128 to a larger value.

### Runtime errors

#### 1. `nncase.simulator.k230.sc: not found`

Q：Compiling kmodel is fine, but when inferring, the error "nncase.simulator.k230.sc: not found" occurs.

A：You need to check whether the versions of nncase and nncase-kpu are the same.

```shell
root@a52f1cacf581:/mnt# pip list | grep nncase
nncase 2.1.1.20230721
nncase-kpu 2.1.1.20230721
```

If inconsistent, install the same version of the Python package ``pip install nncase==x.x.x.x nncase-kpu==x.x.x.x``

### Runtime error on k230 development board

#### Q1. `data.size_bytes() == size = false (bool)`

A: The above situation is usually caused by an error in the input data file of the app inference, which does not match the model input shape or the model input type. Especially when pre-processing is configured, you need to check ` input_shape` and `input_type ` of input data, after adding pre-processing operation, relevant nodes are added to the model, and the input node will also be changed. If   `input_shape `, `input_type `are different from the original model, the newly configured `shape `, `type` should be used to generate input data.

#### Q2.Throwing `std::bad_alloc` exception

A: Usually it is caused by memory allocation failure, you can do the following troubleshooting.

- Check whether the generated kmodel exceeds the current available memory.
- Check whether the generated kmodel exceeds the current available system memory.
