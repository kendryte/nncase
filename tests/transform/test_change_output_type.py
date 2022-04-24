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


def _make_module(in_channel, out_channel, kernel_size):
    class ConvModule(torch.nn.Module):
        def __init__(self):
            super(ConvModule, self).__init__()
            self.conv2d = torch.nn.Conv2d(in_channel, out_channel, kernel_size)

        def forward(self, x):
            x = self.conv2d(x)

            return x

    return ConvModule()


in_shapes = [
    ([1, 3, 24, 24], 3, 3)
]

overwrite_cfgs = [
    """
case: 
    compile_opt:
        is_fpga: false
        dump_asm: true
        dump_ir: true
        dump_quant_error: true
        dump_import_op_range: false
        quant_type: 'int8'
        w_quant_type: 'int8'
        output_type: 'int8'
        output_range: [-6, 6]
        use_mse_quant_w: true
        quant_method: "no_clip"
    """,
    """
    case:
        compile_opt:
            is_fpga: false
            dump_asm: true
            dump_ir: true
            dump_quant_error: true
            dump_import_op_range: false
            quant_type: 'uint8'
            w_quant_type: 'uint8'
            output_type: 'uint8'
            output_range: []
            use_mse_quant_w: true
            quant_method: "no_clip"
    """,
    """
    case:
        compile_opt:
            is_fpga: false
            dump_asm: true
            dump_ir: true
            dump_quant_error: false
            dump_import_op_range: false
            quant_type: 'uint8'
            w_quant_type: 'uint8'
            output_type: 'float32'
            output_range: [6,6]
            use_mse_quant_w: true
            quant_method: "no_clip"
    """
]


@pytest.mark.parametrize('in_shape, out_channel, kernel_size', in_shapes)
@pytest.mark.parametrize('overwrite_cfg', overwrite_cfgs, ids=["int8", "uint8", "float32"])
def test_change_output_type(in_shape, out_channel, kernel_size, overwrite_cfg, request):
    module = _make_module(in_shape[1], out_channel, kernel_size)

    runner = OnnxTestRunner(request.node.name, overwrite_configs=overwrite_cfg)
    model_file = runner.from_torch(module, in_shape)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_change_output_type.py'])
