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
"""System test: test add"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel
import pytest
import os
import tensorflow as tf
import numpy as np
import sys
import test_util


def _make_module(in_shape, v_shape):
    class AddModule(tf.Module):
        def __init__(self):
            super(AddModule).__init__()
            self.v = tf.constant(np.random.rand(*v_shape).astype(np.float32))

        @tf.function(input_signature=[tf.TensorSpec(in_shape, tf.float32)])
        def __call__(self, x):
            outs = []
            outs.append(x + self.v)
            outs.append(x - self.v)
            outs.append(x * self.v)
            outs.append(x / self.v)
            outs.append(tf.minimum(x, self.v))
            outs.append(tf.maximum(x, self.v))
            return outs
    return AddModule()


@pytest.fixture
def module1():
    return _make_module([3, 64, 3], [3])


@pytest.fixture
def module2():
    return _make_module([3, 64, 3], [64, 1])


@pytest.fixture
def module3():
    return _make_module([3, 64, 3], [3, 64, 3])


@pytest.fixture
def module4():
    return _make_module([8, 6, 16, 3], [6, 16, 1])


@pytest.mark.parametrize('module', ['module1', 'module2', 'module3', 'module4'])
def test_binary(module, request):
    module_inst = request.getfixturevalue(module)
    test_util.test_tf_module('test_binary.' + module, module_inst, ['cpu'])


if __name__ == "__main__":
    pytest.main(['-vv', 'test_binary.py'])
