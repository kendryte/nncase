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
import numpy as np

def _make_module(in_shape, epsilon):

    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, in_shape)

    initializers = []
    scale = helper.make_tensor("scale",
                               TensorProto.FLOAT,
                               dims=in_shape[1:2],
                               vals=np.random.randn(in_shape[1],).astype(np.float32).flatten().tolist())
    initializers.append(scale)

    bias = helper.make_tensor("bias",
                              TensorProto.FLOAT,
                              dims=in_shape[1:2],
                              vals=np.random.randn(in_shape[1],).astype(np.float32).flatten().tolist())
    initializers.append(bias)

    if epsilon is None:
        node = onnx.helper.make_node('InstanceNormalization',
                                     inputs=['input', 'scale', 'bias'],
                                     outputs=['output'])
    else:
        node = onnx.helper.make_node('InstanceNormalization',
                                     inputs=['input', 'scale', 'bias'],
                                     outputs=['output'],
                                     epsilon=epsilon)

    graph_def = helper.make_graph([node], 'test-model', [input], [output], initializer=initializers)
    model_def = helper.make_model(graph_def, producer_name='kendryte')

    return model_def

in_shapes = [
    [1, 3, 56, 56]
]

epsilons = [
    None,
    1e-2
]

@pytest.mark.parametrize('in_shape', in_shapes)
@pytest.mark.parametrize('epsilon', epsilons)
def test_instancenorm(in_shape, epsilon, request):
    model_def = _make_module(in_shape, epsilon)

    runner = OnnxTestRunner(request.node.name)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)

if __name__ == "__main__":
    pytest.main(['-vv', 'test_instancenorm.py'])
