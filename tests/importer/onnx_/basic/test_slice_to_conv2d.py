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


def _make_module():

    class FocusModule(torch.nn.Module):
        def __init__(self):
            super(FocusModule, self).__init__()

        def forward(self, feat):
            patch_top_left = feat[..., ::2, ::2]
            patch_top_right = feat[..., ::2, 1::2]
            patch_bot_left = feat[..., 1::2, ::2]
            patch_bot_right = feat[..., 1::2, 1::2]
            x = torch.cat(
                (patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), 1)
            return x

    return FocusModule()


in_shapes = [
    [1, 3, 224, 224],
    [1, 3, 112, 224]
]

out_channels = [
    3,
]


@pytest.mark.parametrize('in_shape', in_shapes)
def test_slice_to_conv2d(in_shape, request):
    module = _make_module()
    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_torch(module, in_shape)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_slice_to_conv2d.py'])
