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


def _make_module():
    class Focus(torch.nn.Module):
        def __init__(self):
            super(Focus, self).__init__()

        def forward(self, x):
            s0 = x[:, :, ::2, :]
            s0 = s0[:, :, :, ::2]

            s1 = x[:, :, 1::2, :]
            s1 = s1[:, :, :, 0::2]

            s2 = x[:, :, ::2, :]
            s2 = s2[:, :, :, 1::2]

            s3 = x[:, :, 1::2, :]
            s3 = s3[:, :, :, 1::2]

            return torch.cat([s0, s1, s2, s3], 1)

    return Focus()


shapes = [
    [1, 3, 640, 640]
]


@pytest.mark.parametrize('shape', shapes)
def test_focus(shape, request):
    module = _make_module()

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_torch(module, shape)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', __file__, '-s'])
