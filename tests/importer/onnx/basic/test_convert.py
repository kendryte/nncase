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
"""System test: test gather"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import pytest
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from onnx_test_runner import OnnxTestRunner


def _make_module(in_shape, in_type, out_type):
    input = helper.make_tensor_value_info('input', in_type, in_shape)
    output = helper.make_tensor_value_info('output', out_type, in_shape)
    node = onnx.helper.make_node(
        'Cast',
        inputs=['input'],
        outputs=['output'],
        to=out_type
    )
    graph_def = helper.make_graph(
        [node],
        'test-cast-model',
        [input],
        [output]
    )
    return helper.make_model(graph_def, producer_name='kendryte')


in_shapes_in_types_out_types = [
    ([8, 3, 12, 3], TensorProto.FLOAT16, TensorProto.FLOAT),
    ([8, 3, 12, 3], TensorProto.FLOAT, TensorProto.FLOAT16),
    ([8, 3, 12, 3], TensorProto.FLOAT, TensorProto.UINT8),
    ([8, 3, 12, 3], TensorProto.FLOAT, TensorProto.INT32),
]


@pytest.mark.parametrize('in_shape,in_type,out_type', in_shapes_in_types_out_types)
def test_convert(in_shape, in_type, out_type, request):
    model_def = _make_module(in_shape, in_type, out_type)
    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_convert.py'])
