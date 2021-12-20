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
"""System test: test binary"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import os
import pytest
import caffe
import numpy as np
from caffe import layers as L
from caffe_test_runner import CaffeTestRunner


def _make_module(model_path, n, i_channels, i_size, operation):
    ns = caffe.NetSpec()
    ns.data1 = L.Input(name="data1", input_param={
                       "shape": {"dim": [n, i_channels, i_size[0], i_size[1]]}})
    ns.data2 = L.Input(name="data2", input_param={
                       "shape": {"dim": [n, i_channels, i_size[0], i_size[1]]}})
    ns.data3 = L.Input(name="data3", input_param={
                       "shape": {"dim": [n, i_channels, i_size[0], i_size[1]]}})
    ns.data4 = L.Input(name="data4", input_param={
                       "shape": {"dim": [n, i_channels, i_size[0], i_size[1]]}})
    ns.ele = L.Eltwise(ns.data1, ns.data2, ns.data3, ns.data4, name="ele",
                       eltwise_param=dict(operation=operation))

    with open(os.path.join(model_path, 'test.prototxt'), 'w') as f:
        f.write(str(ns.to_proto()))

    net = caffe.Net(f.name, caffe.TEST)

    net.save(os.path.join(model_path, 'test.caffemodel'))


n = [
    1
]

i_channels = [
    3
]

i_sizes = [
    [224, 224]
]

operations = [
    0
]


@pytest.mark.parametrize('n', n)
@pytest.mark.parametrize('i_channel', i_channels)
@pytest.mark.parametrize('i_size', i_sizes)
@pytest.mark.parametrize('operation', operations)
def test_norm(n, i_channel, i_size, operation, request):
    overwrite_cfg = """
case: 
  preprocess_opt:
    - name: preprocess
      values:
        - true
    - name: swapRB
      values:
        - false
    - name: input_shape
      values:
        - [1,3,224,224]
    - name: mean
      values:
        - [123,114,109]
    - name: std
      values:
        - [2,2,2]
    - name: input_range
      values:
        - [0,255]
    - name: input_type
      values:
        - uint8
    - name: input_layout
      values:
        - NCHW
    - name: output_layout
      values:
        - NCHW
    - name: letter_value
      values:
        - 0.
"""

    runner = CaffeTestRunner(request.node.name, overwrite_configs=overwrite_cfg)
    model_path = runner.case_dir
    _make_module(model_path, n, i_channel, i_size, operation)
    model_file = [os.path.join(model_path, 'test.prototxt'),
                  os.path.join(model_path, 'test.caffemodel')]
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_norm.py'])
