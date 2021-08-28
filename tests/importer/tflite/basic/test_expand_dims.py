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
import tensorflow as tf
from tensorflow import keras
from tflite_test_runner import TfliteTestRunner


def _make_model(in_shape):
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(4, 3, activation='relu', input_shape=in_shape[1:]))
    return model


in_shapes = [
    [3, 28, 28]
]


@pytest.mark.parametrize('in_shape', in_shapes)
def test_expand_dims(in_shape, request):
    model = _make_model(in_shape)

    runner = TfliteTestRunner(request.node.name)
    model_file = runner.from_keras(model)
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', 'test_expand_dims.py'])
