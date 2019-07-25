`ncc -i <input format> [--dataset <calibration dataset path>] [--postprocess <dataset postprocess>] <input path> <output path>`

- `-i` Input format

| value | description |
|-------|------------------ |
|tflite|`.tflite` TFLite model

- `--inference-type` Inference type

| value | description |
|-------|------------------ |
|uint8| Use quantized kernels (default)
|float| Use float kernels

- `--dataset` Dataset path, **required** when inference type`uint8`.