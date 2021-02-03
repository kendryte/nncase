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


@pytest.fixture
def module():
    class AddModule(tf.Module):

        def __init__(self):
            super(AddModule, self).__init__()
            self.v = tf.constant([9., -2., -7.])

        @tf.function(input_signature=[tf.TensorSpec([1, 2, 3], tf.float32)])
        def __call__(self, x):
            return x + self.v
    return AddModule()


def test_add(module):
    test_util.test_tf_module('test_add', module, ['cpu'])


if __name__ == "__main__":
    pytest.main(['-vv', 'test_add.py'])
