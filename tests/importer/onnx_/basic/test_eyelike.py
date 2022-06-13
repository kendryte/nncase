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


def _make_module(in_shape, off_diagonal_offset):
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, in_shape)
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, in_shape)

    eyelike = onnx.helper.make_node(
        'EyeLike',
        inputs=['x'],
        outputs=['y'],
        k=off_diagonal_offset
    )

    graph_def = helper.make_graph(
        [eyelike],
        'test-model',
        [x],
        [y]
    )

    op = onnx.OperatorSetIdProto()
    op.version = 9
    model_def = helper.make_model(graph_def, producer_name='kendryte', opset_imports=[op])

    return model_def


in_shapes = [
    [3, 5],
    [5, 5],
    [5, 3]
]

off_diagonal_offsets = [
    0,
    -1,
    1,
    -2,
    2,
    -3,
    3
]

@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('off_diagonal_offset', off_diagonal_offsets)
def test_eyelike(in_shape, off_diagonal_offset, request):
    model_def = _make_module(in_shape, off_diagonal_offset)
    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_eyelike.py'])
