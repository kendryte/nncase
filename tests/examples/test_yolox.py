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

import pytest
import os
import subprocess
import sys
import shlex


def test_yolox(request):
    os.chdir('examples/yolox')
    # print(os.environ)
    my_env = os.environ.copy()
    # run cpu float
    ret = subprocess.run(
        shlex.split('python tools/compile.py model/yolox_nano_224.onnx yolox_nano_224.kmodel --legacy'), env=my_env)
    ret.check_returncode()
    ret = subprocess.run(
        shlex.split('python tools/simulate.py yolox_nano_224.kmodel ../20classes_yolo/images/dog.bmp --no_display'), env=my_env)
    ret.check_returncode()
    # run k210 quant
    ret = subprocess.run(
        shlex.split('python tools/compile.py model/yolox_nano_224.onnx yolox_nano_224_quant.kmodel --imgs_dir ../20classes_yolo/images/ --legacy --target k210'), env=my_env)
    ret.check_returncode()
    ret = subprocess.run(
        shlex.split('python tools/simulate.py yolox_nano_224_quant.kmodel ../20classes_yolo/images/dog.bmp --no_display'), env=my_env)
    ret.check_returncode()
    os.chdir('../..')


if __name__ == "__main__":
    pytest.main(['-vv', 'test_yolox.py'])
