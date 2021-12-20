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
"""System test: test permute"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import os
import pytest
import caffe
import numpy as np
from caffe import layers as L
from caffe_test_runner import CaffeTestRunner


def _make_module(model_path, in_shape, order):
    ns = caffe.NetSpec()
    ns.data = L.Input(name="data", input_param={"shape": {"dim": in_shape}})
    ns.perm = L.Permute(ns.data, name="permute", order=order)

    with open(os.path.join(model_path, 'test.prototxt'), 'w') as f:
        f.write(str(ns.to_proto()))

    net = caffe.Net(f.name, caffe.TEST)

    net.save(os.path.join(model_path, 'test.caffemodel'))


in_shapes = [
    [3, 2],
    [64, 3],
    [5, 9, 32],
    [8, 3, 64, 3]
]

orders = [
    [1, 0],
    [0, 2, 1],
    [1, 0, 2],
    [2, 0, 1],
    [2, 1, 0],
    [0, 2, 1, 3],
    [0, 1, 3, 2],
    [3, 1, 0, 2],
    [3, 2, 0, 1],
    [3, 2, 1, 0]
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('order', orders)
def test_permute(in_shape, order, request):
    if len(order) == len(in_shape):
        runner = CaffeTestRunner(request.node.name)
        model_path = runner.case_dir
        _make_module(model_path, in_shape, order)
        model_file = [os.path.join(model_path, 'test.prototxt'),
                      os.path.join(model_path, 'test.caffemodel')]
        runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_permute.py'])
