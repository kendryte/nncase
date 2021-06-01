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
"""System test: test 20 classes yolo"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel
import pytest
import os
import tensorflow as tf
import numpy as np
import sys
import test_util

def test_20classes_yolo(request):
    tflite = os.path.join(os.path.dirname(__file__), '../../../../examples/20classes_yolo/model/20classes_yolo.tflite')
    test_util.test_tflite(request.node.name, tflite, ['cpu', 'k210', 'k510'])


if __name__ == "__main__":
    pytest.main(['-vv', 'test_20classes_yolo.py'])
