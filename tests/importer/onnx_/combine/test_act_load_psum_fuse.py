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


def _make_module(in_shape):

    input_1 = helper.make_tensor_value_info('input_1', TensorProto.FLOAT, in_shape)
    input_2 = helper.make_tensor_value_info('input_2', TensorProto.FLOAT, in_shape)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, in_shape)
    initializers = []

    # add1
    add1 = onnx.helper.make_node(
        'Add',
        inputs=['input_1', 'input_2'],
        outputs=['add1'],
    )

    # batchnorm1
    scale1 = helper.make_tensor(
        'scale1',
        TensorProto.FLOAT,
        dims=in_shape[1:2],
        vals=np.random.randn(in_shape[1],).astype(np.float32).flatten().tolist()
    )
    initializers.append(scale1)

    bias1 = helper.make_tensor(
        'bias1',
        TensorProto.FLOAT,
        dims=in_shape[1:2],
        vals=np.random.randn(in_shape[1],).astype(np.float32).flatten().tolist()
    )
    initializers.append(bias1)

    mean1 = helper.make_tensor(
        'mean1',
        TensorProto.FLOAT,
        dims=in_shape[1:2],
        vals=np.random.randn(in_shape[1],).astype(np.float32).flatten().tolist()
    )
    initializers.append(mean1)

    var1 = helper.make_tensor(
        'var1',
        TensorProto.FLOAT,
        dims=in_shape[1:2],
        vals=np.random.rand(in_shape[1],).astype(np.float32).flatten().tolist()
    )
    initializers.append(var1)

    batchnorm1 = onnx.helper.make_node(
        'BatchNormalization',
        inputs=['add1', 'scale1', 'bias1', 'mean1', 'var1'],
        outputs=['batchnorm1']
    )

    # conv2d
    weight_shape = [in_shape[1], in_shape[1], 1, 1]
    weight = helper.make_tensor(
        'weight',
        TensorProto.FLOAT,
        dims=weight_shape,
        vals=np.random.randn(*weight_shape).astype(np.float32).flatten().tolist())
    initializers.append(weight)

    conv2d = onnx.helper.make_node(
        'Conv',
        inputs=['batchnorm1', 'weight'],
        outputs=['conv2d'],
        kernel_shape=[1, 1],
        pads=[0, 0, 0, 0],
    )

    # add2
    add2 = onnx.helper.make_node(
        'Add',
        inputs=['add1', 'conv2d'],
        outputs=['add2'],
    )

    # batchnorm2
    scale2 = helper.make_tensor(
        'scale2',
        TensorProto.FLOAT,
        dims=in_shape[1:2],
        vals=np.random.randn(in_shape[1],).astype(np.float32).flatten().tolist()
    )
    initializers.append(scale2)

    bias2 = helper.make_tensor(
        'bias2',
        TensorProto.FLOAT,
        dims=in_shape[1:2],
        vals=np.random.randn(in_shape[1],).astype(np.float32).flatten().tolist()
    )
    initializers.append(bias2)

    mean2 = helper.make_tensor(
        'mean2',
        TensorProto.FLOAT,
        dims=in_shape[1:2],
        vals=np.random.randn(in_shape[1],).astype(np.float32).flatten().tolist()
    )
    initializers.append(mean2)

    var2 = helper.make_tensor(
        'var2',
        TensorProto.FLOAT,
        dims=in_shape[1:2],
        vals=np.random.rand(in_shape[1],).astype(np.float32).flatten().tolist()
    )
    initializers.append(var2)

    batchnorm2 = onnx.helper.make_node(
        'BatchNormalization',
        inputs=['add2', 'scale2', 'bias2', 'mean2', 'var2'],
        outputs=['output']
    )

    graph_def = helper.make_graph([add1, batchnorm1, conv2d, add2, batchnorm2],
                                  'test-model', [input_1, input_2], [output], initializer=initializers)
    model_def = helper.make_model(graph_def, producer_name='kendryte')

    return model_def


in_shapes = [
    [1, 32, 56, 56],
    [1, 64, 56, 56],
    [1, 128, 56, 56],
    [1, 256, 56, 56]
]


@pytest.mark.parametrize('in_shape', in_shapes)
def test_act_load_psum_fuse(in_shape, request):
    cfg = '''
    [target]

    [target.cpu]
    eval = false
    infer = false

    [target.k510]
    eval = false
    infer = true

    [target.k230]
    eval = false
    infer = false
    '''

    model_def = _make_module(in_shape)

    runner = OnnxTestRunner(request.node.name, overwrite_configs=cfg)
    model_file = runner.from_onnx_helper(model_def)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_act_load_psum_fuse.py'])
