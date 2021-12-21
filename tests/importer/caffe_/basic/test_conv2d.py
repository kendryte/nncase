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
"""System test: test conv2d"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import os
import pytest
import caffe
import numpy as np
from caffe import layers as L
from caffe_test_runner import CaffeTestRunner


def _make_module(model_path, n, i_channels, i_size, k_size, o_channels, strides, padding, dilations):
    ns = caffe.NetSpec()
    ns.data = L.Input(name="data", input_param={
                      "shape": {"dim": [n, i_channels, i_size[0], i_size[1]]}})
    ns.conv = L.Convolution(ns.data, name="conv2d", kernel_size=k_size, num_output=o_channels,
                            stride=strides, pad=padding, dilation=dilations, weight_filler=dict(type='xavier'))

    with open(os.path.join(model_path, 'test.prototxt'), 'w') as f:
        f.write(str(ns.to_proto()))

    net = caffe.Net(f.name, caffe.TEST)
    for l in net.layers:
        for b in l.blobs:
            if np.count_nonzero(b.data) == 0:
                b.data[...] = np.random.randn(*b.data.shape)

    net.save(os.path.join(model_path, 'test.caffemodel'))


n = [
    1,
    3
]

i_channels = [
    3,
    4
]

i_sizes = [
    [28, 28],
]

k_sizes = [
    1,
    3,
    5
]

o_channels = [
    1,
    4
]

strides = [
    1,
    5
]

paddings = [
    1,
    2
]

dilations = [
    1,
    2
]


@pytest.mark.parametrize('n', n)
@pytest.mark.parametrize('i_channel', i_channels)
@pytest.mark.parametrize('i_size', i_sizes)
@pytest.mark.parametrize('k_size', k_sizes)
@pytest.mark.parametrize('o_channel', o_channels)
@pytest.mark.parametrize('stride', strides)
@pytest.mark.parametrize('padding', paddings)
@pytest.mark.parametrize('dilation', dilations)
def test_conv2d(n, i_channel, i_size, k_size, o_channel, stride, padding, dilation, request):
    runner = CaffeTestRunner(request.node.name)
    model_path = runner.case_dir
    _make_module(model_path, n, i_channel, i_size, k_size, o_channel, stride, padding, dilation)
    model_file = [os.path.join(model_path, 'test.prototxt'),
                  os.path.join(model_path, 'test.caffemodel')]
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_conv2d.py'])
