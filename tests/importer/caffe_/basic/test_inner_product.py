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
"""System test: test inner product"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import os
import pytest
import caffe
import numpy as np
from caffe import layers as L
from caffe_test_runner import CaffeTestRunner


def _make_module(model_path, in_shape, num_output, axis, transpose):
    ns = caffe.NetSpec()
    ns.data = L.Input(name="data", input_param={"shape": {"dim": in_shape}})
    ns.ip = L.InnerProduct(ns.data, name="ip", num_output=num_output,
                           weight_filler=dict(type='xavier'), axis=axis, transpose=transpose)

    with open(os.path.join(model_path, 'test.prototxt'), 'w') as f:
        f.write(str(ns.to_proto()))

    net = caffe.Net(f.name, caffe.TEST)
    for l in net.layers:
        for b in l.blobs:
            if np.count_nonzero(b.data) == 0:
                b.data[...] = np.random.randn(*b.data.shape)

    net.save(os.path.join(model_path, 'test.caffemodel'))


in_shapes = [
    [1, 24],
    [3, 1, 36],
    [10, 2, 48]
]

num_outputs = [
    32,
    64
]

axes = [
    0,
    1,
    2
]

transposes = [
    True,
    False
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('num_output', num_outputs)
@pytest.mark.parametrize('axis', axes)
@pytest.mark.parametrize('transpose', transposes)
def test_inner_product(in_shape, num_output, axis, transpose, request):
    if axis < len(in_shape):
        runner = CaffeTestRunner(request.node.name)
        model_path = os.path.join(os.getcwd(), 'tests_output',
                                  request.node.name.replace('[', '_').replace(']', '_'))
        _make_module(model_path, in_shape, num_output, axis, transpose)
        model_file = [os.path.join(model_path, 'test.prototxt'),
                      os.path.join(model_path, 'test.caffemodel')]
        runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_inner_product.py'])
