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
from onnx import helper, OperatorSetIdProto
from onnx import AttributeProto, TensorProto, GraphProto
from onnx_test_runner import OnnxTestRunner
import numpy as np


def _make_module(in_shape, op_version: int):
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, in_shape)
    initializers = []

    # x*1
    Mul_9 = helper.make_node('Mul', ['input', 'Constant_0'], ['Mul_9'])
    Constant_0 = helper.make_tensor('Constant_0', TensorProto.FLOAT, dims=[1], vals=[100])
    initializers.append(Constant_0)

    # (x*1) + 3
    Add_14 = helper.make_node('Add', ['Mul_9', 'Constant_1'], ['tmp_4'])
    Constant_1 = helper.make_tensor('Constant_1', TensorProto.FLOAT, dims=[1], vals=[20])
    initializers.append(Constant_1)

    # clip((x*1) + 3 , 0 , 6)
    Clip_2 = helper.make_node('Clip', ['tmp_4', 'Constant_2', 'Constant_3'], ['relu6_2.tmp_0'])
    Constant_2 = helper.make_tensor('Constant_2', TensorProto.FLOAT, dims=[], vals=[0])
    Constant_3 = helper.make_tensor('Constant_3', TensorProto.FLOAT, dims=[], vals=[6])
    initializers.extend([Constant_2, Constant_3])

    # clip((x*1) + 3 , 0 , 6) * 0.16
    Mul_10 = helper.make_node('Mul', ['relu6_2.tmp_0', 'Constant_4'], ['Mul_10'])
    Constant_4 = helper.make_tensor('Constant_4', TensorProto.FLOAT, dims=[1], vals=[1 / 6])
    initializers.append(Constant_4)

    # clip((x*1) + 3 , 0 , 6) * 0.16 + 0.
    Add_15 = helper.make_node('Add', ['Mul_10', 'Constant_5'], ['output'])
    Constant_5 = helper.make_tensor('Constant_5', TensorProto.FLOAT, dims=[1], vals=[0])
    initializers.append(Constant_5)

    graph_def = helper.make_graph([Mul_9, Add_14, Clip_2, Mul_10, Add_15],
                                  'test-model', [input], [output], initializer=initializers)
    op = OperatorSetIdProto()
    op.version = op_version
    model_def = helper.make_model(graph_def, producer_name='kendryte', opset_imports=[op])

    return model_def


in_shapes = [
    [1, 12, 14, 14],
    [1, 8, 56, 56],
    [1, 8, 56, 56],
]

op_versions = [
    11,
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('op_version', op_versions)
def test_unary_with_clamp(in_shape, op_version, request):
    model_def = _make_module(in_shape, op_version)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_unary_with_clamp.py'])
