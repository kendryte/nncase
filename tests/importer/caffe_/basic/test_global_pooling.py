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
"""System test: test global pooling"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import os
import pytest
import caffe
import numpy as np
from caffe import layers as L
from caffe_test_runner import CaffeTestRunner


def _make_module(model_path, n, i_channels, i_size, pooling):
    ns = caffe.NetSpec()
    ns.data = L.Input(name="data", input_param={
                      "shape": {"dim": [n, i_channels, i_size[0], i_size[1]]}})
    ns.pool = L.Pooling(ns.data, name="pooling", stride=1, pad=0, pool=pooling, global_pooling=True)

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
    8
]

i_sizes = [
    [28, 28],
]

poolings = [
    0,
    1
]


@pytest.mark.parametrize('n', n)
@pytest.mark.parametrize('i_channel', i_channels)
@pytest.mark.parametrize('i_size', i_sizes)
@pytest.mark.parametrize('pooling', poolings)
def test_global_pooling(n, i_channel, i_size, pooling, request):
    runner = CaffeTestRunner(request.node.name)
    model_path = os.path.join(os.getcwd(), 'tests_output',
                              request.node.name.replace('[', '_').replace(']', '_'))
    _make_module(model_path, n, i_channel, i_size, pooling)
    model_file = [os.path.join(model_path, 'test.prototxt'),
                  os.path.join(model_path, 'test.caffemodel')]
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_global_pooling.py'])
