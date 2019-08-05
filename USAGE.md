`ncc <input path> <output path> -i <input format> [--dataset <calibration dataset path>] [--input-mean <input mean>=0.0] [--input-std <input std>=1.0]`

- `-i` Input format

| value | description |
|-------|------------------ |
|tflite|`.tflite` TFLite model

- `--inference-type` Inference type

| value | description |
|-------|------------------ |
|uint8| Use quantized kernels (default)
|float| Use float kernels

- `--dataset` Dataset path, **required** when inference type `uint8`.

- `--input-mean` `--input-std` Normalize input images , `y = (x - input_mean) / input_std`