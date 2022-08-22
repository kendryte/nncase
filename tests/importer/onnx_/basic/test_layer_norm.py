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

from importlib import import_module
import pytest
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from onnx_test_runner import OnnxTestRunner
import numpy as np


def _make_module(in_shape, axis, epsilon):

    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, in_shape)

    initializers = []
    actual_axis = -1 if axis is None else axis
    scale = helper.make_tensor("scale",
                               TensorProto.FLOAT,
                               dims=in_shape[actual_axis:],
                               vals=np.random.randn(*in_shape[actual_axis:]).astype(np.float32).flatten().tolist())
    initializers.append(scale)

    bias = helper.make_tensor("bias",
                              TensorProto.FLOAT,
                              dims=in_shape[actual_axis:],
                              vals=np.random.randn(*in_shape[actual_axis:],).astype(np.float32).flatten().tolist())
    initializers.append(bias)

    if axis is None and epsilon is None:
        node = onnx.helper.make_node('LayerNormalization',
                                     inputs=['input', 'scale', 'bias'],
                                     outputs=['output'])
    elif axis is None:
        node = onnx.helper.make_node('LayerNormalization',
                                     inputs=['input', 'scale', 'bias'],
                                     outputs=['output'],
                                     epsilon=epsilon)
    elif epsilon is None:
        node = onnx.helper.make_node('LayerNormalization',
                                     inputs=['input', 'scale', 'bias'],
                                     outputs=['output'],
                                     axis=axis)
    else:
        node = onnx.helper.make_node('LayerNormalization',
                                     inputs=['input', 'scale', 'bias'],
                                     outputs=['output'],
                                     axis=axis,
                                     epsilon=epsilon)

    graph_def = helper.make_graph([node], 'test-model', [input], [output], initializer=initializers)
    op = onnx.OperatorSetIdProto()
    op.version = 17
    model_def = helper.make_model(graph_def, producer_name='onnx', opset_imports=[op])

    return model_def


in_shapes = [
    [1, 24, 256]
]

axes = [
    None,
    -1,
    2,
    1,
    0
]

epsilons = [
    None,
    1e-2
]


@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('axis', axes)
@pytest.mark.parametrize('epsilon', epsilons)
def test_layer_norm(in_shape, axis, epsilon, request):
    model_def = _make_module(in_shape, axis, epsilon)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_layer_norm.py'])
