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
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from onnx_test_runner import OnnxTestRunner


def _make_module(in_shape, in_type, out_type, op_version):
    attributes_dict = {}

    attributes_dict['to'] = str(out_type) if op_version == 1 else out_type

    input = helper.make_tensor_value_info('input', in_type, in_shape)
    output = helper.make_tensor_value_info('output', out_type, in_shape)
    node = onnx.helper.make_node(
        'Cast',
        inputs=['input'],
        outputs=['output'],
        **attributes_dict
    )

    graph_def = helper.make_graph(
        [node],
        'test-cast-model',
        [input],
        [output]
    )

    op = onnx.OperatorSetIdProto()
    op.version = op_version
    model_def = helper.make_model(graph_def, producer_name='kendryte', opset_imports=[op])
    return model_def


in_shapes_in_types_out_types = [
    ([8, 3, 12, 3], TensorProto.FLOAT16, TensorProto.FLOAT),
    ([8, 3, 12, 3], TensorProto.FLOAT, TensorProto.FLOAT16),
    ([8, 3, 12, 3], TensorProto.FLOAT, TensorProto.UINT8),
    ([8, 3, 12, 3], TensorProto.FLOAT, TensorProto.INT32),
]

op_versions = [
    # op_version 1 is not supported by onnxruntime yet: Could not find an implementation for the node Cast(1)
    # 1,
    6,
    9,
    13
]


@pytest.mark.parametrize('in_shape,in_type,out_type', in_shapes_in_types_out_types)
@pytest.mark.parametrize('op_version', op_versions)
def test_cast(in_shape, in_type, out_type, op_version, request):
    model_def = _make_module(in_shape, in_type, out_type, op_version)
    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_cast.py'])
