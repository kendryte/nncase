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
import torch
import numpy as np
from onnx_test_runner import OnnxTestRunner


def _make_module(model_shape, h_0_shape, c_0_shape):
    class LstmModule(torch.nn.Module):
        def __init__(self):
            super(LstmModule, self).__init__()
            # self.h_0 = torch.from_numpy(np.zeros(h_0_shape).astype(np.float32))
            # self.c_0 = torch.from_numpy(np.zeros(c_0_shape).astype(np.float32))
            self.lstm = torch.nn.LSTM(*model_shape)

        def forward(self, x):
            # return self.lstm(x, (self.h_0, self.c_0))[0]
            return self.lstm(x, )[0]

    return LstmModule()


D = [1]
num_layers = [1, 3, 5]
batch_size = [1]
hidden_size = [5, 8]
input_size = [10, 23]
length = [2, 4]


@pytest.mark.parametrize('D', D)
@pytest.mark.parametrize('num_layers', num_layers)
@pytest.mark.parametrize('batch_size', batch_size)
@pytest.mark.parametrize('hidden_size', hidden_size)
@pytest.mark.parametrize('input_size', input_size)
@pytest.mark.parametrize('length', length)
def test_lstm(D, num_layers, batch_size, hidden_size, input_size, length, request):
    c_0_shapes = [
        D * num_layers, batch_size, hidden_size
    ]

    h_0_shapes = [
        D * num_layers, batch_size, hidden_size
    ]

    inputs_shapes = [
        length, batch_size, input_size
    ]

    model_shapes = [
        input_size, hidden_size, num_layers
    ]

    module = _make_module(model_shapes, h_0_shapes, c_0_shapes)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_torch(module, inputs_shapes)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_lstm.py'])
