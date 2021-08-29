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
"""System test: test batchnorm"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import os
import pytest
import caffe
import numpy as np
from caffe import layers as L
from caffe_test_runner import CaffeTestRunner


def _make_module(model_path, n, i_channels, i_size, eps):
    ns = caffe.NetSpec()
    ns.data = L.Input(name="data", input_param={
                      "shape": {"dim": [n, i_channels, i_size[0], i_size[1]]}})
    ns.bn = L.BatchNorm(ns.data, name="bn", eps=eps)

    with open(os.path.join(model_path, 'test.prototxt'), 'w') as f:
        f.write(str(ns.to_proto()))

    net = caffe.Net(f.name, caffe.TEST)
    for l in net.layers:
        for b in l.blobs:
            if np.count_nonzero(b.data) == 0:
                for i in range(0, b.data.size):
                    b.data[i] = np.random.rand()

    net.save(os.path.join(model_path, 'test.caffemodel'))


n = [
    1,
    3
]

i_channels = [
    3,
    8
]

i_sizes = [
    [28, 28],
]

eps = [
    1e-5,
    1e-4
]


@pytest.mark.parametrize('n', n)
@pytest.mark.parametrize('i_channel', i_channels)
@pytest.mark.parametrize('i_size', i_sizes)
@pytest.mark.parametrize('eps', eps)
def test_batchnorm(n, i_channel, i_size, eps, request):
    runner = CaffeTestRunner(request.node.name)
    model_path = os.path.join(os.getcwd(), 'tests_output',
                              request.node.name.replace('[', '_').replace(']', '_'))
    _make_module(model_path, n, i_channel, i_size, eps)
    model_file = [os.path.join(model_path, 'test.prototxt'),
                  os.path.join(model_path, 'test.caffemodel')]
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_batchnorm.py'])
