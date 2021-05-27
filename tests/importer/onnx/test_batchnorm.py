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
import test_util

def _make_module(num, esp, momentum):

    class BatchNormModule(torch.nn.Module):
        def __init__(self):
            super(BatchNormModule, self).__init__()
            self.batchnorm = torch.nn.BatchNorm2d(num, esp, momentum)

        def forward(self, x):
            x = self.batchnorm(x)
            return x

    return BatchNormModule()

in_shapes = [
    [1, 2, 16, 16],
    [1, 8, 224, 224]
]

epses = [
    0.00001,
    0.00005
]

momentums = [
    0.1,
    0.9
]

@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('eps', epses)
@pytest.mark.parametrize('momentum', momentums)
def test_batchnorm(in_shape, eps, momentum, request):
    module = _make_module(in_shape[1], eps, momentum)

    test_util.test_onnx_module(request.node.name, module, in_shape, ['cpu', 'k210', 'k510'])

if __name__ == "__main__":
    pytest.main(['-vv', 'test_batchnorm.py'])