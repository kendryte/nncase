# Copyright 2019-2021 Canaan Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import pytest
from onnx_test_runner import OnnxTestRunner


def test_efficientnet_lite4_11(request):

    overwrite_file = '''
case: # case的配置，应该是一个多层次的
  preprocess_opt:
    - name: preprocess
      values:
        - true
    - name: swapRB
      values:
        - false
    - name: input_shape
      values:
        - []
    - name: mean
      values:
        - [0,0,0]
    - name: std
      values:
        - [1,1,1]
    - name: input_range
      values:
        - [0,255]
    - name: input_type
      values:
        - float32
    - name: input_layout
      values:
        - NHWC
    - name: model_layout
      values:
        - NHWC
    - name: output_layout
      values:
        - NHWC
    - name: letterbox_value
      values:
        - 0.
  importer_opt:
    kwargs: null
  compile_opt:
    is_fpga: false
    dump_asm: true
    dump_ir: true
    dump_quant_error: false
    dump_import_op_range: false
    quant_type: 'uint8'
    w_quant_type: 'uint8'
    use_mse_quant_w: true
    quant_method: "l2"

  generate_inputs:
    name: generate_imagenet_dataset
    kwargs: 
      dir_path: "ImageNet/ImageNet_1000_first"
    numbers: 1
    batch_size: 100
  generate_calibs:
    name: generate_imagenet_dataset
    kwargs:
      dir_path: "ImageNet/ImageNet_1000_first"
    numbers: 1
    batch_size: 10
  generate_dump_range_data:
    name: generate_imagenet_dataset
    kwargs:
      dir_path: "ImageNet/ImageNet_1000_first"
    numbers: 1
    batch_size: 1
  eval:
    - name: target
      values:
        - cpu
    - name: ptq
      values:
        - false
  infer:
    - name: target
      values:
        - cpu
    - name: ptq
      values:
        - false
        - true
judge:
  common: &judge_common
    simarity_name:  top1
    threshold: 0.01
    log_hist: true
    matchs: null
  specifics:
    - matchs:
        target: [cpu]
        ptq: true
      threshold: 1
    '''
    runner = OnnxTestRunner(request.node.name, ['cpu', 'k510'], overwrite_configs=overwrite_file)
    model_file = 'onnx-models/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx'
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_efficientnet-lite4-11.py'])
