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
"""System test: test reduce"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel
import pytest
import os
import tensorflow as tf
import numpy as np
import sys
import test_util


def _make_module(in_shape, axis, keep_dims=False):
    class ReduceModule(tf.Module):
        def __init__(self):
            super(ReduceModule).__init__()

        @tf.function(input_signature=[tf.TensorSpec(in_shape, tf.float32)])
        def __call__(self, x):
            outs = []
            outs.append(tf.reduce_min(x, axis=axis, keepdims=keep_dims))
            outs.append(tf.reduce_max(x, axis=axis, keepdims=keep_dims))
            outs.append(tf.reduce_mean(x, axis=axis, keepdims=keep_dims))
            outs.append(tf.reduce_sum(x, axis=axis, keepdims=keep_dims))
            return outs
    return ReduceModule()


@pytest.fixture
def module1():
    return _make_module([3], [0])


@pytest.fixture
def module2():
    return _make_module([64, 3], [0])

@pytest.fixture
def module3():
    return _make_module([64, 3], [1])

@pytest.fixture
def module4():
    return _make_module([64, 3], [0, 1])


@pytest.fixture
def module5():
    return _make_module([3, 64, 3], [1, 2])


@pytest.fixture
def module6():
    return _make_module([8, 6, 16, 3], [1, 3])


@pytest.fixture
def module7():
    return _make_module([8, 6, 16, 3], [2, 3, 1])


@pytest.fixture
def module8():
    return _make_module([8, 6, 16, 3], [0, 2, 1], keep_dims=True)


@pytest.mark.parametrize('module', ['module1', 'module2', 'module3', 'module4', 'module5', 'module6', 'module7', 'module8'])
def test_reduce(module, request):
    module_inst = request.getfixturevalue(module)
    test_util.test_tf_module('test_reduce.' + module, module_inst, ['cpu'])


if __name__ == "__main__":
    pytest.main(['-vv', 'test_reduce.py'])
