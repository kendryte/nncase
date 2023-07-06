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


def _make_module(case):
    class EinsumModule(torch.nn.Module):
        def __init__(self):
            super(EinsumModule, self).__init__()
            self.v = torch.from_numpy(np.random.rand(*case[1]).astype(np.float32))

        def forward(self, x):
            outs = []
            outs.append(torch.einsum(case[2], x, self.v))
            return outs

    return EinsumModule()


cases = [
    [[3], [4], "i,j->ij"],
    [[2, 4, 6], [6, 4, 3], "ibh,hnd->ibnd"],
    [[4, 2, 5, 6], [3, 2, 5, 6], "ibnd,jbnd->bnij"],
    [[2, 5, 4, 6], [6, 2, 5, 3], "bnij,jbnd->ibnd"],
    [[5, 2, 3, 4], [6, 3, 4], "ibnd,hnd->ibh"]
]


@pytest.mark.parametrize('case', cases)
def test_einsum(case, request):
    module = _make_module(case)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_torch(module, case[0], 12)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(
        ['-vv', 'test_einsum.py'])
