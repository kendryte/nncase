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


def _make_module(shape):
    class GridSampleModule(torch.nn.Module):
        def __init__(self):
            super(GridSampleModule, self).__init__()
            self.grid = torch.from_numpy(np.random.random(shape[1]).astype(np.float32) * 2 - 1)

        def forward(self, x):
            outs = []
            outs.append(torch.nn.functional.grid_sample(x, self.grid))
            return outs

    return GridSampleModule()


shapes = [
    [[8, 32, 80, 80], [8, 300, 4, 2]],
    [[8, 32, 40, 40], [8, 300, 4, 2]],
    [[8, 32, 20, 20], [8, 300, 4, 2]],
]


@pytest.mark.parametrize('shape', shapes)
def test_grid_sample(shape, request):
    module = _make_module(shape)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_torch(module, shape[0], 16)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(
        ['-vv', 'test_grid_sample.py'])
