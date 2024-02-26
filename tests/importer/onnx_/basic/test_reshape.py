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
from onnx_test_runner import OnnxTestRunner


def _make_module(in_shape):

    class ReshapeModule(torch.nn.Module):
        def __init__(self):
            super(ReshapeModule, self).__init__()
            self.n = in_shape[0]
            self.c = in_shape[1]
            self.h = in_shape[2]
            self.w = in_shape[3]

        def forward(self, x):
            out = []
            out.append(torch.reshape(x, (self.n, self.c * self.h * self.w)))
            out.append(torch.reshape(x, (self.n, self.c, self.h * self.w)))
            out.append(torch.reshape(x, (self.n, self.c * self.h, self.w)))
            out.append(torch.reshape(x, (self.n * self.c, self.h * self.w)))
            out.append(torch.reshape(x, (self.n * self.c, self.h, self.w)))
            out.append(torch.reshape(x, (self.n * self.c * self.h, self.w)))
            out.append(torch.reshape(x, (-1, self.w)))
            return out

    return ReshapeModule()


in_shapes = [
    [1, 4, 60, 72],
    [1, 3, 224, 224],
    [3, 4, 5, 6]
]


@pytest.mark.parametrize('in_shape', in_shapes)
def test_reshape(in_shape, request):
    module = _make_module(in_shape)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_torch(module, in_shape)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_reshape.py'])
