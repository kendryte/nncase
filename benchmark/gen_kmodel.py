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
import os
import torch
import torchvision as tv
import numpy as np
import sys
import nncase
import requests
import onnxsim

TEMP_DIR = "tmp"
MODEL_DIR = "models"


def _download(url, name, in_shape):
    filename = os.path.join(MODEL_DIR, "source", name + ".onnx")
    if not os.path.exists(filename):
        req = requests.get(url)
        with open(filename, 'wb') as file:
            for chunk in req.iter_content(100000):
                file.write(chunk)
        try:
            onnx_model, check = onnxsim.simplify(filename, input_shapes=in_shape)
            assert check, "Simplified ONNX model could not be validated"
            with open(filename, 'wb') as file:
                file.write(onnx_model)
        except:
            print("WARN: optimize onnx failed")
    with open(filename, "rb") as file:
        return file.read()


def _make_module():
    name = "minist"
    url = "https://media.githubusercontent.com/media/onnx/models/master/vision/classification/mnist/model/mnist-8.onnx"
    onnx_model = _download(url, name, [1, 1, 28, 28])
    target = "cpu"

    # import
    compile_options = nncase.CompileOptions()
    compile_options.target = target
    compile_options.dump_dir = TEMP_DIR
    compile_options.benchmark_only = True
    compiler = nncase.Compiler(compile_options)

    import_options = nncase.ImportOptions()
    import_options.input_layout = "NCHW"
    import_options.output_layout = "NCHW"
    compiler.import_onnx(onnx_model, import_options)

    # compile
    compiler.compile()
    kmodel = compiler.gencode_tobytes()
    with open(os.path.join(MODEL_DIR, target, name + ".kmodel"), 'wb') as f:
        f.write(kmodel)


if __name__ == "__main__":
    _make_module()
