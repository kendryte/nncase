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
"""System test: test lstm"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import os
import pytest
import caffe
import numpy as np
from caffe import layers as L
from caffe_test_runner import CaffeTestRunner


def _make_module(model_path, in1_shape, time_step, batch_size, num_output):
    ns = caffe.NetSpec()
    ns.data1 = L.Input(name="data1", input_param={"shape": {"dim": in1_shape}})
    ns.data2 = L.ContinuationIndicator(name="data2", continuation_indicator_param={
                                       "time_step": time_step, "batch_size": batch_size})
    ns.lstm = L.LSTM(ns.data1, ns.data2, name="lstm", recurrent_param={"num_output": num_output})

    with open(os.path.join(model_path, 'test.prototxt'), 'w') as f:
        f.write(str(ns.to_proto()))

    net = caffe.Net(f.name, caffe.TEST)
    for l in net.layers:
        for b in l.blobs:
            if np.count_nonzero(b.data) == 0:
                b.data[...] = np.random.randn(*b.data.shape)

    net.save(os.path.join(model_path, 'test.caffemodel'))


in1_shapes = [
    [2, 1, 28],
]

time_steps = [
    2
]

batch_sizes = [
    1
]

num_outputs = [
    28,
]


@pytest.mark.parametrize('in1_shape', in1_shapes)
@pytest.mark.parametrize('time_step', time_steps)
@pytest.mark.parametrize('batch_size', batch_sizes)
@pytest.mark.parametrize('num_output', num_outputs)
def test_lstm(in1_shape, time_step, batch_size, num_output, request):
    runner = CaffeTestRunner(request.node.name)
    model_path = os.path.join(os.getcwd(), 'tests_output',
                              request.node.name.replace('[', '_').replace(']', '_'))
    _make_module(model_path, in1_shape, time_step, batch_size, num_output)
    model_file = [os.path.join(model_path, 'test.prototxt'),
                  os.path.join(model_path, 'test.caffemodel')]
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_lstm.py'])
